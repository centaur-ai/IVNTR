"""An approach that learns world model in the raw-state space, during testing,
use shooting to solve the task. We could choose to use "causal" of "vanilla".
."""

import os
import logging
import functools
from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn
import torch.optim
import dill as pkl
from gym.spaces import Box

from predicators import utils
from predicators.envs.satellites import SatellitesSimpleEnv
from predicators.approaches.grader import GRADER
from predicators.approaches.grader.utils import load_config
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.gnn_approach import GNNApproach
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.option_model import create_option_model
from predicators.ground_truth_models import get_dummy_nsrts
from predicators.settings import CFG
from predicators.planning import task_plan_grounding
from predicators.structs import Action, Array, Dataset, DummyOption, \
    GroundAtom, Object, ParameterizedOption, Predicate, State, Task, Type, \
    _Option

class MBShootingApproach(GNNApproach):
    """Trains and uses a GNN world model for online planning."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._sorted_options = sorted(self._initial_options,
                                      key=lambda o: o.name)
        # dummy nsrt helps us:
        # 1. Define the necessary preconditions
        # 2. Define the GT sampler
        self._dummy_nsrts = get_dummy_nsrts(CFG.env, self._initial_predicates,
                                 self._initial_options)
        # check the options and nsrts are one-one mapping
        assert len(self._sorted_options) == len(self._dummy_nsrts)
        self._option2nsrt = {}
        for nsrt in self._dummy_nsrts:
            nsrt_option = nsrt.option
            assert nsrt_option in self._sorted_options
            self._option2nsrt[nsrt_option] = nsrt
        self._option_model = create_option_model(CFG.option_model_name)
        assert os.path.exists(CFG.grader_config), "Config not exist!"
        config = load_config(config_path=CFG.grader_config)
        self.agent_config = config["AGENT"]
        self._agent = GRADER(self.agent_config)
        self._mse_loss = torch.nn.MSELoss()
        self._shooting_horizon = self.agent_config['planner']['horizon']
        self._num_shoots = self.agent_config['planner']['popsize']
        self.possible_ent_names = [] # for input node rank
        self.option_obj2action = {}
        self.action2option_obj = {}
        

    def _generate_data_from_dataset(
        self, dataset: Dataset
    ) -> Tuple[List[Tuple[State, Set[GroundAtom], State, Set[GroundAtom], _Option, str]], List]:
        data = []
        # In this approach, we never learned any NSRTs, so we just call
        # segment_trajectory() to segment the given dataset.
        segmented_trajs = [
            segment_trajectory(traj, self._initial_predicates)
            for traj in dataset.trajectories
        ]
        episode_idx = 0
        for segment_traj, ll_traj in zip(segmented_trajs,
                                         dataset.trajectories):
            if not ll_traj.is_demo:
                continue
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            for segment in segment_traj:
                state = segment.states[0]  # the segment's initial state
                atoms = segment.init_atoms  # the segment's initial atoms
                state_ = segment.states[-1]  # the segment's final state
                atoms_ = segment.final_atoms
                action = segment.get_option()  # the segment's option
                data.append((state, atoms, state_, atoms_, goal, action, episode_idx))
            episode_idx += 1
        return data
    
    def _setup_state_fields(
        self, data) -> None:
        obj_types_set = set()
        nullary_predicates_set = set()
        unary_predicates_set = set()
        binary_predicates_set = set()
        obj_attrs_set = set()

        # Go through the data, identifying the types/predicates/attributes.
        for state, atoms, _, _, goal, _, _ in data:
            for atom in atoms | goal:
                arity = atom.predicate.arity
                assert arity <= 2, "Predicates with arity > 2 are not supported"
                if arity == 0:
                    nullary_predicates_set.add(atom.predicate)
                elif arity == 1:
                    unary_predicates_set.add(atom.predicate)
                elif arity == 2:
                    binary_predicates_set.add(atom.predicate)
            for obj in state:
                obj_types_set.add(f"type_{obj.type.name}")
                for feat in obj.type.feature_names:
                    obj_attrs_set.add(f"feat_{feat}")
        self._nullary_predicates = sorted(nullary_predicates_set)
        obj_types = sorted(obj_types_set)
        unary_predicates = sorted(unary_predicates_set)
        binary_predicates = sorted(binary_predicates_set)
        obj_attrs = sorted(obj_attrs_set)

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        # Initialize input node features.
        self._node_feature_to_index = {}
        index = 0
        for obj_type in obj_types:
            self._node_feature_to_index[obj_type] = index
            self._node_is_rot.append(False)
            index += 1
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[unary_predicate] = index
            index += 1
            self._node_is_rot.append(False)
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[G(unary_predicate)] = index
            self._node_is_rot.append(False)
            index += 1
        for obj_attr in obj_attrs:
            self._node_feature_to_index[obj_attr] = index
            if "_q" in obj_attr:
                self._node_is_rot.append(True)
            else:
                self._node_is_rot.append(False)
            index += 1
        assert self.agent_config['env_params']['obj_centric_feat_dim'] == \
                        len(self._node_feature_to_index)
        # Initialize input edge features.
        self._edge_feature_to_index = {}
        index = 0
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[binary_predicate] = index
            self._edge_is_rot.append(False)
            index += 1
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[R(binary_predicate)] = index
            self._edge_is_rot.append(False)
            index += 1
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[G(binary_predicate)] = index
            self._edge_is_rot.append(False)
            index += 1
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[G(R(binary_predicate))] = index
            self._edge_is_rot.append(False)
            index += 1
        assert self.agent_config['env_params']['rel_feat_dim'] == \
                        len(self._edge_feature_to_index)

    def _steup_action_fields(self, data) -> None:
        action_id = 0
        for s, _, _, _, _, action, _ in data:
            base_option = action.parent
            act_obj = []
            for obj in action.objects:
                act_obj.append(obj.name) # use name
            option_obj = (base_option, tuple(act_obj))
            if option_obj not in self.option_obj2action:
                self.option_obj2action[option_obj] = action_id
                action_id += 1
            for obj in list(s):
                if obj.name not in self.possible_ent_names:
                    self.possible_ent_names.append(obj.name)
        self.action2option_obj = {v: k for k, v in self.option_obj2action.items()}
        return

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        data = self._generate_data_from_dataset(dataset)
        # state construction
        self._setup_state_fields(data)
        # action construction
        self._steup_action_fields(data)
        # Store Transition
        # Note that agent can't expore the env, all the offline data are stored
        last_episode = 0
        for state, atoms, state_, atoms_, goal, action, episode in data:
            flattened_state = self._vectorize_input(state=state, atoms=atoms, goal=goal)
            flattened_state_ = self._vectorize_input(state=state_, atoms=atoms_, goal=goal)
            flattened_action = self._vectorize_action(action)
            # if episode > last_episode:
            #     logging.info(f"Episode {episode} collection finished, training the agent...")
            #     self._agent.train()
            #     last_episode = episode
            self._agent.store_transition([flattened_state, flattened_action, flattened_state_])
        self._agent.train()
        save_path = utils.get_approach_save_path_str()
        info = {
            "node_feature_to_index": self._node_feature_to_index,
            "edge_feature_to_index": self._edge_feature_to_index,
            "option_obj2action": self.option_obj2action,
            "action2option_obj": self.action2option_obj,
            "possible_ent_names": self.possible_ent_names,
        }
        self._agent.save_model(info)
        with open(f"{save_path}.wm", "wb") as f:
            pkl.dump(info, f)

    def _setup_output_specific_fields(
        self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom],
                               _Option]]
    ) -> None:
        # Go through the data, identifying the maximum number of option
        # objects and parameters.
        max_option_objects = 0
        max_option_params = 0
        for _, _, _, option in data:
            assert len(option.params.shape) == 1
            max_option_objects = max(max_option_objects, len(option.objects))
            max_option_params = max(max_option_params, option.params.shape[0])
        self._max_option_objects = max_option_objects
        self._max_option_params = max_option_params

    def _vectorize_input(self, state: State, atoms: Set[GroundAtom],
                               goal: Set[GroundAtom]) -> Tuple[Dict, Dict]:
        all_objects = list(state)
        node_to_object = {}
        object_to_node = {}
        for obj in all_objects:
            obj_name = obj.name
            obj_id = self.possible_ent_names.index(obj_name)
            node_to_object[obj_id] = obj
            object_to_node[obj] = obj_id

        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        obj_cent_feat = np.zeros((num_objects, num_node_features))

        ## Add node features for obj types.
        for obj in state:
            obj_index = object_to_node[obj]
            type_index = self._node_feature_to_index[f"type_{obj.type.name}"]
            obj_cent_feat[obj_index, type_index] = 1

        ## Add node features for unary atoms.
        for atom in atoms:
            if atom.predicate.arity != 1:
                continue
            obj_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[atom.predicate]
            obj_cent_feat[obj_index, atom_index] = 1

        ## Add node features for unary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 1:
                continue
            obj_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[G(atom.predicate)]
            obj_cent_feat[obj_index, atom_index] = 1

        ## Add node features for state.
        for obj in state:
            obj_index = object_to_node[obj]
            for feat, val in zip(obj.type.feature_names, state[obj]):
                feat_index = self._node_feature_to_index[f"feat_{feat}"]
                if "theta" in feat:
                    obj_cent_feat[obj_index, feat_index] = val / np.pi
                else:
                    obj_cent_feat[obj_index, feat_index] = val
        obj_cent_feat = obj_cent_feat.flatten()

        # Add edges (one between each pair of objects) and edge features.
        rel_feat_all = np.zeros(
            (num_objects, num_objects, num_edge_features))

        ## Add edge features for binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[atom.predicate]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            rel_feat_all[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[R(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            rel_feat_all[obj1_index, obj0_index, pred_index] = 1

        ## Add edge features for binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            rel_feat_all[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(R(atom.predicate))]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            rel_feat_all[obj1_index, obj0_index, pred_index] = 1

        # Organize into expected representation.
        # Note that we already have Rev in the edge features, so we don't need to
        # distinguish the direction of the edges.
        rel_feat_non_self = []
        for i in range(num_objects):
            for j in range(num_objects):
                if i == j:
                    continue
                rel_feat_non_self.append(rel_feat_all[i, j])
        rel_feat_non_self = np.array(rel_feat_non_self).flatten()

        vectorized_feat = np.concatenate([obj_cent_feat, rel_feat_non_self])

        return vectorized_feat

    def _vectorize_action(self, action: _Option) -> Dict:
        action_array = np.zeros(self.agent_config['env_params']['action_dim'])
        base_option = action.parent
        names = []
        for obj in action.objects:
            names.append(obj.name)
        action_id = self.option_obj2action[(base_option, tuple(names))]
        action_array[action_id] = 1
        return action_array

    def _graphify_single_target(self, target) -> Dict:
        pass

    def _extract_output_from_graph(
        self, graph_output: Dict, object_to_node: Dict
    ) -> Tuple[ParameterizedOption, List[Object], Array]:
        """The output is a parameterized option from self._sorted_options,
        discrete object arguments, and continuous arguments."""
        node_to_object = {v: k for k, v in object_to_node.items()}
        type_to_node = defaultdict(set)
        for obj, node in object_to_node.items():
            type_to_node[obj.type.name].add(node)
        # Extract parameterized option and continuous parameters.
        onehot_output, params = np.split(graph_output["globals"],
                                         [len(self._sorted_options)])
        param_opt = self._sorted_options[np.argmax(onehot_output)]
        # Pad and clip parameters.
        params = params[:param_opt.params_space.shape[0]]
        params = params.clip(param_opt.params_space.low,
                             param_opt.params_space.high)
        # Extract objects, making sure to enforce the typing constraints.
        objects = []
        for i, obj_type in enumerate(param_opt.types):
            scores = graph_output["nodes"][:, i]
            allowed_idxs = type_to_node[obj_type.name]
            for j in range(len(scores)):
                if j not in allowed_idxs:
                    scores[j] = float("-inf")  # set its score to be really bad
            if np.max(scores) == float("-inf"):
                # If all scores are -inf, we failed to select an object.
                raise ApproachFailure(
                    "GNN option policy could not select an object")
            objects.append(node_to_object[np.argmax(scores)])
        return param_opt, objects, params

    @classmethod
    def get_name(cls) -> str:
        return "mb_shooting"

    def load(self, online_learning_cycle) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}.wm", "rb") as f:
            info = pkl.load(f)
        self._node_feature_to_index = info["node_feature_to_index"]
        self._edge_feature_to_index = info["edge_feature_to_index"]
        self.option_obj2action = info["option_obj2action"]
        self.action2option_obj = {v: k for k, v in self.option_obj2action.items()}
        self.possible_ent_names = info["possible_ent_names"]
        self._agent.load_model(info)

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        return self._solve_with_shooting(task, timeout)

    def _predict(self, state: State, atoms: Set[GroundAtom],
                    goal: Set[GroundAtom], task: Task,
                    action_list: list, ground_nsrts: list) -> int:
        vectorized_state = self._vectorize_input(state, atoms, goal)
        vectorized_state = vectorized_state[None]
        actions = []
        env = SatellitesSimpleEnv()
        env.render_state_plt(state, task, save_path="init.png")
        goal_reached = False
        for i in range(self._num_shoots):
            if goal_reached:
                break
            actions.append([])
            curr_actions = action_list
            curr_vectorized_state = vectorized_state
            for t in range(self._shooting_horizon):
                if len(curr_actions) == 0:
                    # no applicable actions
                    break
                action = self._rng.choice(curr_actions)
                # action = 0 # debug
                actions[-1].append(action)
                action_array = np.eye(self._agent.action_dim)[action]
                action_array = action_array[None]
                vectorized_state_next = self._agent.planner.model.predict(curr_vectorized_state, action_array) \
                                        + vectorized_state
                state_next = self._unvectorize_state_satellites(vectorized_state_next[0], list(state), task.init)
                env.render_state_plt(state_next, task, save_path=f"{i}_{t}_a{action}.png")
                if task.goal_holds(state_next):
                    goal_reached = True
                    break
                else:
                    # update the current applicable actions
                    # seems that using all initially applicable actions is better
                    # curr_actions = []
                    # atoms = utils.abstract(state_next, self._initial_predicates)
                    # for ground_nsrt in ground_nsrts:
                    #     if ground_nsrt.preconditions.issubset(atoms):
                    #         base_option = ground_nsrt.option
                    #         ent_ids = []
                    #         for obj in ground_nsrt.objects:
                    #             ent_ids.append(self.possible_ent_names.index(obj.name))
                    #         grounded_action = GroundedAction(base_option, ent_ids)
                    #         curr_actions.append(self._grounded_action_list.index(grounded_action))
                    curr_vectorized_state = self._vectorize_input(state_next, atoms, goal)[None]

        action = actions[-1][0]
        return action
    
    def _unvectorize_state_blocks(self, vectorized_state: Array, objs: List[Object]) \
        -> State:
        data = {}
        for i, obj_name in enumerate(self.possible_ent_names):
            node_feat = vectorized_state[i*len(self._node_feature_to_index):
                              (i+1)*len(self._node_feature_to_index)]
            data_x = node_feat[self._node_feature_to_index["feat_pose_x"]]
            data_y = node_feat[self._node_feature_to_index["feat_pose_y"]]
            data_z = node_feat[self._node_feature_to_index["feat_pose_z"]]
            data_held = node_feat[self._node_feature_to_index["feat_held"]]
            data_finger = node_feat[self._node_feature_to_index["feat_fingers"]]
            data_r = node_feat[self._node_feature_to_index["feat_color_r"]]
            data_g = node_feat[self._node_feature_to_index["feat_color_g"]]
            data_b = node_feat[self._node_feature_to_index["feat_color_b"]]
            for obj in objs:
                assert obj.name in self.possible_ent_names
                if obj.name == obj_name:
                    if 'rob' in obj_name:
                        array_data = np.array([data_x, data_y, data_z, data_finger])
                        array_data = array_data.clip(0, 1)
                        data[obj] = array_data
                    else:
                        assert 'block' in obj_name
                        array_data = np.array([data_x, data_y, data_z, data_held, data_r, data_g, data_b])
                        array_data = array_data.clip(0, 1)
                        data[obj] = array_data
        return State(data)
    
    def _unvectorize_state_satellites(self, vectorized_state: Array, \
                            objs: List[Object], init_state: State) \
                            -> State:
        # hacking function for SatellitesSimpleEnv, for model based shooting
        data = {}
        for i, obj_name in enumerate(self.possible_ent_names):
            node_feat = vectorized_state[i*len(self._node_feature_to_index):
                              (i+1)*len(self._node_feature_to_index)]
            for obj in objs:
                assert obj.name in self.possible_ent_names
                if obj.name == obj_name:
                    if 'sat' in obj_name:
                        # changable part
                        data_x = node_feat[self._node_feature_to_index["feat_x"]].clip(0, 1)
                        data_y = node_feat[self._node_feature_to_index["feat_y"]].clip(0, 1)
                        # unnormalize theta
                        data_theta = node_feat[self._node_feature_to_index["feat_theta"]] * np.pi
                        data_read_obj_id = round(node_feat[self._node_feature_to_index["feat_read_obj_id"]])
                        data_is_calibrated = round(node_feat[self._node_feature_to_index["feat_is_calibrated"]])
                        # fixed part
                        data_insturment = init_state.get(obj, "instrument")
                        data_calibration_obj_id = init_state.get(obj, "calibration_obj_id")
                        data_shoots_chem_x = init_state.get(obj, "shoots_chem_x")
                        data_shoots_chem_y = init_state.get(obj, "shoots_chem_y")
                        if 'feat_view_clear' in self._node_feature_to_index:
                            data_view_clear = round(node_feat[self._node_feature_to_index["feat_view_clear"]])
                            data[obj] = {
                                'x': data_x,
                                'y': data_y,
                                'theta': data_theta,
                                'instrument': data_insturment,
                                'calibration_obj_id': data_calibration_obj_id,
                                'is_calibrated': data_is_calibrated,
                                'read_obj_id': data_read_obj_id,
                                'shoots_chem_x': data_shoots_chem_x,
                                'shoots_chem_y': data_shoots_chem_y,
                                'view_clear': data_view_clear,
                            }
                        else:
                            data[obj] = {
                                'x': data_x,
                                'y': data_y,
                                'theta': data_theta,
                                'instrument': data_insturment,
                                'calibration_obj_id': data_calibration_obj_id,
                                'is_calibrated': data_is_calibrated,
                                'read_obj_id': data_read_obj_id,
                                'shoots_chem_x': data_shoots_chem_x,
                                'shoots_chem_y': data_shoots_chem_y
                            }
                    else:
                        assert 'obj' in obj_name
                        # changable part
                        data_x = node_feat[self._node_feature_to_index["feat_x"]].clip(0, 1)
                        data_y = node_feat[self._node_feature_to_index["feat_y"]].clip(0, 1)
                        # fixed part
                        data_id = init_state.get(obj, "id")
                        data_has_chem_x = init_state.get(obj, "has_chem_x")
                        data_has_chem_y = init_state.get(obj, "has_chem_y")
                        data[obj] = {
                            'x': data_x,
                            'y': data_y,
                            'id': data_id,
                            'has_chem_x': data_has_chem_x,
                            'has_chem_y': data_has_chem_y
                        }
        return utils.create_state_from_dict(data)
    
    def _criterion(self, output: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        """Define the criterion function for passing into train_model()."""
        pass

    def _global_criterion(self, output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        """Define the global criterion function for passing into
        train_model()."""
        pass

    def _add_output_specific_fields_to_save_info(self, info: Dict) -> None:
        """Given a dict of info to be saved, add output-specific fields to
        it."""
        pass

    def _load_output_specific_fields_from_save_info(self, info: Dict) -> None:
        """Given a dict of saved info, load output-specific fields from it."""
        pass

    def _solve_with_shooting(self, task: Task,
                             timeout: int) -> Callable[[State], Action]:
        cur_option = DummyOption
        logging.info("Opt_Obj to Actions ID:")
        for opt_obj, action in enumerate(self.option_obj2action):
            logging.info(f"{opt_obj}: {action}")
        logging.info("Ent Id to Name:")
        for i, ent_name in enumerate(self.possible_ent_names):
            logging.info(f"{i}: {ent_name}")
        memory: Dict = {}  # optionally updated by predict()
        # Keep trying until the timeout.
        name2obj = {}
        for obj in list(task.init):
            name2obj[obj.name] = obj
        init_atoms = utils.abstract(task.init, self._initial_predicates)
        objects = set(task.init)
        ground_nsrts, _ = task_plan_grounding(
            init_atoms, objects, self._dummy_nsrts, allow_noops=True)

        def _policy(state: State) -> Action:
            nonlocal cur_option
            atoms = utils.abstract(state, self._initial_predicates)
            curr_applicabel_actions = []
            for ground_nsrt in ground_nsrts:
                if ground_nsrt.preconditions.issubset(atoms):
                    base_option = ground_nsrt.option
                    ent_names = []
                    for obj in ground_nsrt.objects:
                        ent_names.append(obj.name)
                    option_obj = (base_option, tuple(ent_names))
                    if option_obj not in self.option_obj2action:
                        # do not take never seen actions
                        continue
                    curr_applicabel_actions.append(self.option_obj2action[option_obj])
            if cur_option is DummyOption or cur_option.terminal(state):
                # get applicable grounded action
                # sample applicable action using WM pred
                action_idx = self._predict(
                    state, atoms, task.goal, task, curr_applicabel_actions, ground_nsrts)
                option, obj_names = self.action2option_obj[action_idx]
                objects = [name2obj[obj_name] for obj_name in obj_names]
                dummy_nsrt = self._option2nsrt[option]
                grounded_nsrt = dummy_nsrt.ground(objects)
                cur_option = grounded_nsrt.sample_option(state, goal=task.goal, rng=self._rng)
                if not cur_option.initiable(state):
                    raise ApproachFailure(
                        "MB-shooting policy chose a non-initiable option")
            act = cur_option.policy(state)
            return act
        
        return _policy
