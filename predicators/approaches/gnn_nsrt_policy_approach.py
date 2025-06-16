"""An approach that trains a GNN mapping states, atoms, and goals to
dummy NSRTs, with GT sampler."""

import os
import time
import logging
import itertools
from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import dill as pkl
import torch
import torch.nn
import torch.optim
from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.gnn_approach import GNNApproach
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.option_model import create_option_model
from predicators.ground_truth_models import get_dummy_nsrts
from predicators.settings import CFG
from predicators.gnn.gnn_utils import get_single_model_prediction, normalize_graph
from predicators.structs import Action, Array, Dataset, DummyOption, \
    GroundAtom, NSRT, ParameterizedOption, Predicate, State, Task, Type, \
    _Option
from predicators.nsrt_learning.sampler_learning import _create_neural_sampler

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

class GNNNSRTPolicyApproach(GNNApproach):
    """Trains and uses a goal-conditioned GNN policy that produces options."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._sorted_options = sorted(self._initial_options,
                                      key=lambda o: o.name)
        self._option_model = create_option_model(CFG.option_model_name)
        self._max_option_objects = 0
        self._bce_loss = torch.nn.BCEWithLogitsLoss()
        self._crossent_loss = torch.nn.CrossEntropyLoss()
        # dummy nsrt helps us:
        # 1. Define the necessary preconditions
        # 2. Define the GT sampler
        env_dummy_nsrts = get_dummy_nsrts(CFG.env, self._initial_predicates,
                                    self._initial_options)
        self._dummy_nsrts = set()
        for env_nsrt in env_dummy_nsrts:
            if CFG.ivntr_nsrt_path != '':
                if ".neupi_info" in CFG.ivntr_nsrt_path:
                    logging.info(f"Loading learned NSRT samplers from {CFG.ivntr_nsrt_path}")
                    with open(CFG.ivntr_nsrt_path, "rb") as f:
                        content_info = pkl.load(f)
                    ivntr_dummy_nsrts = content_info["dummy_nsrts"]
                    for ivntr_nsrt in ivntr_dummy_nsrts:
                        if ivntr_nsrt.option == env_nsrt.option:
                            new_nsrt = NSRT(name=env_nsrt.name, parameters=env_nsrt.parameters, preconditions=env_nsrt.preconditions,
                                add_effects=env_nsrt.add_effects, delete_effects=env_nsrt.delete_effects,
                                ignore_effects=env_nsrt.ignore_effects, option=env_nsrt.option,
                                option_vars=env_nsrt.option_vars, _sampler=ivntr_nsrt.sampler)
                            self._dummy_nsrts.add(new_nsrt)
                else:
                    assert ".saved" in CFG.ivntr_nsrt_path
                    logging.info(f"Loading learned NSRT samplers from {CFG.ivntr_nsrt_path} (pts)")
                    parameters = env_nsrt.parameters
                    option = env_nsrt.option
                    name = self._sorted_options.index(option)
                    cls_weight_path = f"{CFG.ivntr_nsrt_path}_sampler_op_Op{name}_cls_weight.pt"
                    cls_info_path = f"{CFG.ivntr_nsrt_path}_sampler_op_Op{name}_cls_info.pt"
                    reg_weight_path = f"{CFG.ivntr_nsrt_path}_sampler_op_Op{name}_reg_weight.pt"
                    reg_info_path = f"{CFG.ivntr_nsrt_path}_sampler_op_Op{name}_reg_info.pt"
                    if os.path.exists(cls_weight_path):
                        sampler = _create_neural_sampler(parameters, option,
                                                cls_weight_path, 
                                                cls_info_path,
                                                reg_weight_path, reg_info_path)
                    else:
                        logging.info(f"Using GT sampler (Null) for NSRT {env_nsrt.name}")
                        sampler = env_nsrt.sampler
                    new_nsrt = NSRT(name=env_nsrt.name, parameters=env_nsrt.parameters, preconditions=env_nsrt.preconditions,
                                add_effects=env_nsrt.add_effects, delete_effects=env_nsrt.delete_effects,
                                ignore_effects=env_nsrt.ignore_effects, option=env_nsrt.option,
                                option_vars=env_nsrt.option_vars, _sampler=sampler)
                    self._dummy_nsrts.add(new_nsrt)
            else:
                logging.info(f"Using GT sampler for NSRT {env_nsrt.name}")
                self._dummy_nsrts.add(env_nsrt)
        # check the options and nsrts are one-one mapping
        # assert len(self._sorted_options) == len(self._dummy_nsrts)
        self._option2nsrt = {}
        for nsrt in self._dummy_nsrts:
            nsrt_option = nsrt.option
            assert nsrt_option in self._sorted_options
            self._option2nsrt[nsrt_option] = nsrt
        for option in self._sorted_options:
            if option not in self._option2nsrt:
                logging.warning(f"Option {option} has no dummy NSRT, adding GT for it")
                for env_nsrt in env_dummy_nsrts:
                    if env_nsrt.option == option:
                        self._option2nsrt[option] = env_nsrt
                        self._dummy_nsrts.add(env_nsrt)
                        break

    def _generate_data_from_dataset(
        self, dataset: Dataset
    ) -> List[Tuple[State, Set[GroundAtom], Set[GroundAtom], _Option]]:
        data = []
        # In this approach, we never learned any NSRTs, so we just call
        # segment_trajectory() to segment the given dataset.
        segmented_trajs = [
            segment_trajectory(traj, self._initial_predicates)
            for traj in dataset.trajectories
        ]
        for segment_traj, ll_traj in zip(segmented_trajs,
                                         dataset.trajectories):
            if not ll_traj.is_demo:
                continue
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            for segment in segment_traj:
                state = segment.states[0]  # the segment's initial state
                atoms = segment.init_atoms  # the segment's initial atoms
                target = segment.get_option()  # the segment's option
                data.append((state, atoms, goal, target))
        return data

    def _setup_output_specific_fields(
        self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom],
                               _Option]]
    ) -> None:
        # Go through the data, identifying the maximum number of option
        # objects and parameters.
        max_option_objects = 0
        for _, _, _, option in data:
            assert len(option.params.shape) == 1
            max_option_objects = max(max_option_objects, len(option.objects))
        self._max_option_objects = max_option_objects

    def _graphify_single_target(self, target: _Option, graph_input: Dict,
                                object_to_node: Dict) -> Dict:
        # First, copy over all unchanged fields.
        graph_target = {
            "n_node": graph_input["n_node"],
            "n_edge": graph_input["n_edge"],
            "edges": graph_input["edges"],
            "senders": graph_input["senders"],
            "receivers": graph_input["receivers"],
        }
        # Next, set up the target node features. The target is an _Option.
        object_mask = np.zeros((len(object_to_node), self._max_option_objects),
                               dtype=np.int64)
        for i, obj in enumerate(target.objects):
            object_mask[object_to_node[obj], i] = 1
        graph_target["nodes"] = object_mask
        # Finally, set up the target globals.
        option_index = self._sorted_options.index(target.parent)
        onehot_target = np.zeros(len(self._sorted_options))
        onehot_target[option_index] = 1
        # no param learning required
        graph_target["globals"] = onehot_target
        return graph_target

    def _criterion(self, output: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        if self._max_option_objects == 0:
            return torch.tensor(0.0)
        return self._bce_loss(output, target)

    def _global_criterion(self, output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        onehot_loss = self._crossent_loss(output,
                                          target.argmax(dim=1))
        return onehot_loss

    def _add_output_specific_fields_to_save_info(self, info: Dict) -> None:
        info["max_option_objects"] = self._max_option_objects

    def _load_output_specific_fields_from_save_info(self, info: Dict) -> None:
        self._max_option_objects = info["max_option_objects"]

    def _extract_output_from_graph(
        self, graph_output: Dict, object_to_node: Dict
    ) -> List[Tuple[ParameterizedOption, List[object], float]]:
        """Return a ranked list of (selected_nsrt, objects, combined_score) tuples.
        - Convert logits to probabilities using sigmoid.
        - Compute a combined score = global_score + total_node_score.
        - Return all possible combinations (option, objects) ranked by combined score.
        """
        node_to_object = {v: k for k, v in object_to_node.items()}
        type_to_node = defaultdict(set)
        for obj, node in object_to_node.items():
            type_to_node[obj.type.name].add(node)

        # Convert global logits to probabilities
        global_logits = graph_output["globals"]  # shape [num_options]
        global_probs = sigmoid(global_logits)

        # Sort options by their global probability in descending order
        global_scores = [(opt, p) for opt, p in zip(self._sorted_options, global_probs)]
        global_scores.sort(key=lambda x: x[1], reverse=True)

        # Convert node logits to probabilities
        node_logits = graph_output["nodes"]  # shape [num_nodes, num_params]
        node_probs = sigmoid(node_logits)

        ranked_results = []
        for param_opt, global_score in global_scores:
            selected_nsrt = self._option2nsrt[param_opt]
            param_types = param_opt.types

            # For each parameter type, find candidate nodes
            node_scores_for_opt = []
            for i, obj_type in enumerate(param_types):
                scores = node_probs[:, i].copy()
                allowed_idxs = list(type_to_node[obj_type.name])
                if not allowed_idxs:
                    node_scores_for_opt = []
                    break

                # Mask out invalid nodes by setting score = -inf
                for j in range(len(scores)):
                    if j not in allowed_idxs:
                        scores[j] = float("-inf")

                candidate_nodes = [(j, s) for j, s in enumerate(scores) if s > float("-inf")]
                if not candidate_nodes:
                    node_scores_for_opt = []
                    break
                node_scores_for_opt.append(candidate_nodes)

            if not node_scores_for_opt:
                # No valid combinations for this option
                continue

            # Compute all combinations of nodes for this option
            all_combinations = itertools.product(*node_scores_for_opt)

            # Rank combinations by sum of node probabilities + global probability
            for combo in all_combinations:
                chosen_nodes = [c[0] for c in combo]
                chosen_scores = [c[1] for c in combo]
                total_node_score = sum(chosen_scores)
                combined_score = global_score + total_node_score
                chosen_objects = [node_to_object[n_idx] for n_idx in chosen_nodes]
                ranked_results.append((selected_nsrt, chosen_objects, combined_score))

        # Sort globally by combined score (descending)
        ranked_results.sort(key=lambda x: x[2], reverse=True)

        return ranked_results
    
    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom], memory: Dict):
        del memory  # unused
        # Get output graph.
        in_graph, object_to_node = self._graphify_single_input(
            state, atoms, goal)
        if CFG.gnn_do_normalization:
            in_graph = normalize_graph(in_graph, self._input_normalizers)
        out_graph = get_single_model_prediction(self._gnn, in_graph, CFG.device)
        if CFG.gnn_do_normalization:
            out_graph = normalize_graph(out_graph,
                                        self._target_normalizers,
                                        invert=True)
        # Extract the output from the output graph.
        return self._extract_output_from_graph(out_graph, object_to_node)
    
    @classmethod
    def get_name(cls) -> str:
        return "gnn_nsrt_policy"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        assert self._gnn is not None, "Learning hasn't happened yet!"
        if CFG.gnn_option_policy_solve_with_shooting:
            return self._solve_with_shooting(task, timeout)
        return self._solve_without_shooting(task)

    def _solve_without_shooting(self, task: Task) -> Callable[[State], Action]:
        # no simulator, one-time feedforward prediction
        cur_option = DummyOption
        goal = task.goal
        memory: Dict = {}  # optionally updated by predict()

        def _policy(state: State) -> Action:
            atoms = utils.abstract(state, self._initial_predicates)
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                possible_choices = self._predict(
                        state, atoms, task.goal, memory)
                try_num = 0
                while True:
                    selected_nsrt, objects, _ = possible_choices[0]
                    ground_nsrt = selected_nsrt.ground(objects)
                    if ground_nsrt.preconditions.issubset(atoms):
                        cur_option = ground_nsrt
                        break
                    else:
                        logging.warning(
                            f"Option failed preconditions {try_num}/{len(possible_choices)} times")
                        possible_choices.pop(0)
                        try_num += 1
                cur_option = ground_nsrt.sample_option(state, goal=goal, rng=self._rng)
                # Just use the mean parameters to ground the option.
                if not cur_option.initiable(state):
                    raise ApproachFailure(
                        "GNN option policy chose a non-initiable option")
            act = cur_option.policy(state)
            return act

        return _policy

    def _solve_with_shooting(self, task: Task,
                             timeout: int) -> Callable[[State], Action]:
        # has simulator, shooting + checking until timeout
        start_time = time.perf_counter()
        goal = task.goal
        memory: Dict = {}  # optionally updated by predict()
        # Keep trying until the timeout.
        tries: int = 0
        all_num_act: int = 0
        total_num_act: int = 0
        while time.perf_counter() - start_time < timeout:
            tries += 1
            all_num_act += total_num_act
            total_num_act = 0
            state = task.init
            plan: List[_Option] = []
            # A single shooting try goes up to the environment's horizon.
            while total_num_act < CFG.horizon:
                if task.goal_holds(state):
                    # We found a plan that achieves the goal under the
                    # option model, so return it.
                    option_policy = utils.option_plan_to_policy(plan)

                    def _policy(s: State) -> Action:
                        try:
                            return option_policy(s)
                        except utils.OptionExecutionFailure as e:
                            raise ApproachFailure(e.args[0], e.info)

                    return _policy
                atoms = utils.abstract(state, self._initial_predicates)
                possible_choices = self._predict(
                        state, atoms, task.goal, memory)
                # Sample an initiable option.
                ground_nsrt = None
                for i in range(CFG.gnn_option_policy_shooting_max_samples):
                    if i >= len(possible_choices):
                        continue
                    selected_nsrt, objects, _ = possible_choices[i]
                    try:
                        ground_nsrt = selected_nsrt.ground(objects)
                    except AssertionError:
                        continue
                    if ground_nsrt.preconditions.issubset(atoms):
                        break
                if ground_nsrt is None:
                    break  # out of the while loop for this shooting try
                try:
                    cur_option = ground_nsrt.sample_option(state, goal=goal, rng=self._rng)
                except utils.OptionExecutionFailure as e:
                    break
                plan.append(cur_option)
                # Use the option model to determine the next state.
                try:
                    state, num_act = \
                        self._option_model.get_next_state_and_num_actions(
                            state, cur_option)
                except utils.EnvironmentFailure:
                    break
                # If num_act is zero, that means that the option is stuck in
                # the state, so we should break to avoid infinite loops.
                # Break early if we have timed out.
                total_num_act += num_act
                if time.perf_counter() - start_time > timeout:
                    break
                if num_act == 0:
                    break
        all_num_act += total_num_act
        logging.info(f"Shooting: {all_num_act} actions with {tries} tries in \
              {time.perf_counter() - start_time} seconds")
        raise ApproachTimeout("Shooting timed out!",
                              info={
                                  "partial_refinements": (None, plan),
                              })
