"""An approach that invents predicates by learning a state autoencoder and a
action model (LatPlan). Here we use the first-order version
@inproceedings{asai2019unsupervised,
  title={Unsupervised grounding of plannable first-order logic representation from images},
  author={Asai, Masataro},
  booktitle={Proceedings of the International Conference on Automated Planning and Scheduling},
  volume={29},
  pages={583--591},
  year={2019}
}
"""

from __future__ import annotations

import os
import copy
import glob
import time
import copy
import wandb
import yaml
import random
import functools
import json
import logging
from z3 import *
from tqdm import tqdm
from gym.spaces import Box
from tqdm import tqdm
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set, Tuple, TypeVar

import numpy as np
import dill as pkl
import torch
import torch.nn
import torch.optim
from torch.multiprocessing import Process, Queue
from torch.utils.data import DataLoader

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.gnn_nsrt_policy_approach import GNNNSRTPolicyApproach
from predicators.gnn.gnn_utils import GraphDictDataset, compute_normalizers, action_graph_batch_collate, \
    normalize_graph
from predicators.gnn.fosae import setup_sae_net, get_single_model_prediction
from predicators.gnn.gnn_utils import GraphDictDataset, compute_normalizers, \
    graph_batch_collate, normalize_graph, train_model
from predicators.gnn.gnn import setup_pointnet_graph_net, setup_graph_net
from predicators.ground_truth_models import get_dummy_nsrts
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.settings import CFG
from predicators.predicate_search_score_functions import _OperatorBeliefScoreFunction
from predicators.structs import Dataset, GroundAtom, GroundAtomTrajectory, LowLevelTrajectoryReward, \
    _Option, ParameterizedOption, DummyPredicate, NeuralPredicate, Predicate, Object, State, Task, Type, \
    Action
from predicators.structs import NSRT, PNAD, GroundAtomTrajectory, \
    LowLevelTrajectory, ParameterizedOption, Predicate, Segment, Task

_Output = TypeVar("_Output")  # a generic type for the output of this GNN

def fosae_crit(output, target):
    return torch.nn.functional.mse_loss(output, target)

def train_val_sae_model(model, train_dataloader, val_dataloader, optimizer, num_epochs, val_freq_sae=10,
                        device=None
                        ):
    best_val_loss = float("inf")
    best_model_dict = None
    # get date and time
    os.makedirs(CFG.approach_dir, exist_ok=True)
    save_path_model = os.path.join(CFG.approach_dir, f"sae_best.pt")
    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        total_loss = 0
        for data in train_dataloader:
            inputs = data['graph_input']
            targets = data['graph_target']
            for key in targets.keys():
                if targets[key] is not None:
                    targets[key] = targets[key].detach()
            if device is not None:
                for key, val in inputs.items():
                    inputs[key] = val.to(device) if val is not None else val
                for key, val in targets.items():
                    targets[key] = val.to(device) if val is not None else val
            output = model(inputs.copy())
            loss = fosae_crit(output, targets["nodes"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch} Train Loss: {total_loss / len(train_dataloader)}")
        if epoch % val_freq_sae == 0:
            model.eval()
            model.to(device)
            total_val_loss = 0
            for data in val_dataloader:
                inputs = data['graph_input']
                targets = data['graph_target']
                for key in targets.keys():
                    if targets[key] is not None:
                        targets[key] = targets[key].detach()
                if device is not None:
                    for key, val in inputs.items():
                        inputs[key] = val.to(device) if val is not None else val
                    for key, val in targets.items():
                        targets[key] = val.to(device) if val is not None else val
                output = model(inputs.copy())
                loss = fosae_crit(output, targets["nodes"])
                total_val_loss += loss.item()
            logging.info(f"Epoch {epoch} Val Loss: {total_val_loss / len(val_dataloader)}")
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_model_dict = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path_model)
    # Put result in the queue
    return best_model_dict, best_val_loss

################################################################################
#                                 Approach                                     #
################################################################################


class FOSAEApproach(GNNNSRTPolicyApproach):
    """Note that this approach is not the official FOSAE, since:
    (1) We don't have the "mixed data", which means the AMA model learning is not possible
    (2) For bilevel planning, the goal is not specified as low-level state (like images), but discrete predicates.
        This makes the "implicit predicate" in the original FOSAE not applicable for search to the goal (as we don't know
        how to expand the implicit graph to lead to the goal)
    (3) Bilevel learning has the explicit action space, making "reconstructing the action" not necessary. LatPlan used
        latent (nullary) action spaces, which is not applicable here.
    Therefore, we just do the sae learning as in the official paper, then we directly use the latent code to do behavior cloning
    like the GNN approach
    """

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # load pred settings
        assert os.path.exists(CFG.fosae_pred_config), "Pred Settings Not Found"
        config_dict = yaml.safe_load(open(CFG.fosae_pred_config, 'r'))
        self.learning_cfg = config_dict["learning"]
        self.model_cfg = config_dict["model"]
        self.inference_cfg = config_dict["inference"]
        # sae model does not reconstruct everything
        self._sae_node_feature_to_index = {}
        self._sae_model = None
        self._ama_gnn = None

    @classmethod
    def get_name(cls) -> str:
        return "fosae_policy"

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates
    
    def _setup_input_fields(
        self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom],
                               _Output]]
    ) -> None:
        obj_types_set = set()
        nullary_predicates_set = set()
        unary_predicates_set = set()
        binary_predicates_set = set()
        obj_attrs_set = set()

        # Go through the data, identifying the types/predicates/attributes.
        for state, atoms, goal, _ in data:
            num_obj = len(list(state))
            assert num_obj <= CFG.fosae_max_n, "Too many objects for this FOSAE model"
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
        self._sae_node_feature_to_index = {}
        index = 0
        sae_index = 0
        for obj_type in obj_types:
            self._sae_node_feature_to_index[obj_type] = sae_index
            self._node_feature_to_index[obj_type] = index
            self._node_is_rot.append(False)
            index += 1
            sae_index += 1
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[unary_predicate] = index
            index += 1
            self._node_is_rot.append(False)
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[G(unary_predicate)] = index
            self._node_is_rot.append(False)
            index += 1
        for obj_attr in obj_attrs:
            if obj_attr in self.model_cfg["sae"]["selected_feat"]:
                self._sae_node_feature_to_index[obj_attr] = sae_index
                sae_index += 1
            if 'pcd' in obj_attr:
                # pcd features will hugely expand the feature space
                for i in range(CFG.blocks_engrave_num_points * 3):
                    self._node_feature_to_index[f"{obj_attr}_{i}"] = index
                    self._node_is_rot.append(False)
                    index += 1
                continue
            if 'norm' in obj_attr:
                # norm features will hugely expand the feature space
                for i in range(CFG.blocks_engrave_num_points * 3):
                    self._node_feature_to_index[f"{obj_attr}_{i}"] = index
                    self._node_is_rot.append(True)
                    index += 1
                continue
            self._node_feature_to_index[obj_attr] = index
            if "_q" in obj_attr:
                self._node_is_rot.append(True)
            else:
                self._node_is_rot.append(False)
            index += 1

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

    def _setup_ama_output_specific_fields(
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

    def _graphify_sae_single_input(self, state: State, atoms: Set[GroundAtom],
                               goal: Set[GroundAtom]) -> Tuple[Dict, Dict]:
        all_objects = list(state)
        node_to_object = dict(enumerate(all_objects))
        object_to_node = {v: k for k, v in node_to_object.items()}
        num_objects = len(all_objects)
        num_node_features = len(self._sae_node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        graph = {}

        # Input globals: nullary predicates in atoms and goal.
        atoms_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        for atom in atoms:
            if atom.predicate.arity != 0:
                continue
            atoms_globals[self._nullary_predicates.index(atom.predicate)] = 1
        goal_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        for atom in goal:
            if atom.predicate.arity != 0:
                continue
            goal_globals[self._nullary_predicates.index(atom.predicate)] = 1
        graph["globals"] = np.r_[atoms_globals, goal_globals]

        # Add nodes (one per object) and node features.
        graph["n_node"] = np.array(num_objects)
        node_features = np.zeros((num_objects, num_node_features))

        ## Add node features for obj types.
        for obj in state:
            obj_index = object_to_node[obj]
            type_index = self._sae_node_feature_to_index[f"type_{obj.type.name}"]
            node_features[obj_index, type_index] = 1

        ## Add node features for unary atoms.
        # for atom in atoms:
        #     if atom.predicate.arity != 1:
        #         continue
        #     obj_index = object_to_node[atom.objects[0]]
        #     atom_index = self._node_feature_to_index[atom.predicate]
        #     node_features[obj_index, atom_index] = 1

        ## Add node features for unary atoms in goal.
        # for atom in goal:
        #     if atom.predicate.arity != 1:
        #         continue
        #     obj_index = object_to_node[atom.objects[0]]
        #     atom_index = self._node_feature_to_index[G(atom.predicate)]
        #     node_features[obj_index, atom_index] = 1

        ## Add node features for state.
        for obj in state:
            obj_index = object_to_node[obj]
            for feat, val in zip(obj.type.feature_names, state[obj]):
                if f"feat_{feat}" in self._sae_node_feature_to_index:
                    feat_index = self._sae_node_feature_to_index[f"feat_{feat}"]
                    if "theta" in feat:
                        node_features[obj_index, feat_index] = val / (2 * np.pi)
                    else:
                        node_features[obj_index, feat_index] = val

        graph["nodes"] = node_features

        # Deal with edge case (pun).
        num_edge_features = max(num_edge_features, 1)

        # Add edges (one between each pair of objects) and edge features.
        all_edge_features = np.zeros(
            (num_objects, num_objects, num_edge_features))

        ## Add edge features for binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[atom.predicate]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            all_edge_features[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[R(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[obj1_index, obj0_index, pred_index] = 1

        ## Add edge features for binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            all_edge_features[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(R(atom.predicate))]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[obj1_index, obj0_index, pred_index] = 1

        # Organize into expected representation.
        adjacency_mat = np.any(all_edge_features, axis=2)
        receivers, senders, edges = [], [], []
        for sender, receiver in np.argwhere(adjacency_mat):
            edge = all_edge_features[sender, receiver]
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)

        n_edge = len(edges)
        graph["edges"] = np.reshape(edges, [n_edge, num_edge_features])
        graph["receivers"] = np.reshape(receivers, [n_edge]).astype(np.int64)
        graph["senders"] = np.reshape(senders, [n_edge]).astype(np.int64)
        graph["n_edge"] = np.reshape(n_edge, [1]).astype(np.int64)

        return graph, object_to_node

    def _graphify_ama_single_output(self, target: _Option, graph_input: Dict,
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
            

    def learn_sae(
        self, dataset: Dataset
    ) -> Tuple[List[GroundAtomTrajectory], Dict[Predicate, float]]:
        """Learn the Neural predicates by Action Effect Martix Identification."""
        logging.info("Constructing NeuPi Data...")
        # 1. Generate sae data from the dataset.
        data = self._generate_data_from_dataset(dataset)
        # 2. Setup the input fields for the sae and ama model
        self._setup_input_fields(data)
        num_feat_per_obj = len(self._sae_node_feature_to_index)
        self._sae_model = setup_sae_net(
            CFG.fosae_max_n,
            num_feat_per_obj,
            self.model_cfg["sae"])
        if os.path.exists(CFG.fosae_sae_model_path):
            self._sae_model.load_state_dict(torch.load(CFG.fosae_sae_model_path))
            logging.info(f"Loaded SAE model from {CFG.fosae_sae_model_path}")
            return data
        # 3. Set up all the graphs, now using *all* the data.
        inputs = [(d[0], d[1], d[2]) for d in data]
        targets = [d[3] for d in data]
        graph_inputs = []
        graph_targets = []
        for (state, atoms, goal), target in zip(inputs, targets):
            graph_input, _ = self._graphify_sae_single_input(
                state=state, atoms=atoms, goal=goal)
            graph_inputs.append(graph_input)
            graph_targets.append(graph_input)
        if CFG.fosae_sae_do_normalization:
            # Update normalization constants. Note that we do this for both
            # the input graph and the target graph.
            self._input_normalizers = compute_normalizers(graph_inputs, 
                                        normalize_nodes=self._node_is_rot, \
                                        normalize_edges=self._edge_is_rot, \
                                        normalize_globals=False)
            tgt_node_indicators = [False] * graph_targets[0]["nodes"].shape[1]
            tgt_edge_indicators = [False] * graph_targets[0]["edges"].shape[1]
            self._target_normalizers = compute_normalizers(graph_targets, 
                                        normalize_nodes=tgt_node_indicators, \
                                        normalize_edges=tgt_edge_indicators, \
                                        normalize_globals=False)
            graph_inputs = [
                normalize_graph(g, self._input_normalizers)
                for g in graph_inputs
            ]
            graph_targets = [
                normalize_graph(g, self._target_normalizers)
                for g in graph_targets
            ]
        # Run training.
        if CFG.fosae_use_validation_set:
            ## Split data, using 10% for validation.
            num_validation = max(1, int(len(inputs) * 0.1))
        else:
            num_validation = 0
        train_inputs = graph_inputs[num_validation:]
        train_targets = graph_targets[num_validation:]
        val_inputs = graph_inputs[:num_validation]
        val_targets = graph_targets[:num_validation]
        train_dataset = GraphDictDataset(train_inputs, train_targets)
        val_dataset = GraphDictDataset(val_inputs, val_targets)
        ## Set up Adam optimizer and dataloaders.
        sae_lr = self.learning_cfg["sae"]["lr"]
        training_bs = self.learning_cfg["sae"]["batch_size"]
        val_bs = self.learning_cfg["sae"]["batch_size_val"]
        optimizer = torch.optim.Adam(self._sae_model.parameters(),
                                     lr=sae_lr)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=training_bs,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      persistent_workers=True,
                                      collate_fn=graph_batch_collate)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=val_bs,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=graph_batch_collate)
        ## Train the model.
        best_model_dict, best_val_loss = train_val_sae_model(
            self._sae_model,
            train_dataloader,
            val_dataloader,
            optimizer,
            self.learning_cfg["sae"]["epochs"],
            device=CFG.device
        )
        self._sae_model.load_state_dict(best_model_dict)
        return data
    
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
    
    def _graphify_ama_single_input(self, state: State, atoms: Set[GroundAtom],
                               goal: Set[GroundAtom]) -> Tuple[Dict, Dict]:
        all_objects = list(state)
        node_to_object = dict(enumerate(all_objects))
        object_to_node = {v: k for k, v in node_to_object.items()}
        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)
        
        sae_graph, _ = self._graphify_sae_single_input(state, atoms, goal)
        with torch.no_grad():
            self._sae_model.eval()
            self._sae_model.to(CFG.device)
            input_x = torch.zeros(1, self._sae_model.max_n, \
                self._sae_model.F).to(CFG.device)
            input_x[0, :num_objects, :] = \
                torch.tensor(sae_graph["nodes"]).to(CFG.device)
            z = self._sae_model.encode(input_x, 1)

        z = z.squeeze(0).detach().cpu().numpy()
        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        graph = {}

        # Input globals: nullary predicates in atoms and goal.
        atoms_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        for atom in atoms:
            if atom.predicate.arity != 0:
                continue
            atoms_globals[self._nullary_predicates.index(atom.predicate)] = 1
        goal_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        for atom in goal:
            if atom.predicate.arity != 0:
                continue
            goal_globals[self._nullary_predicates.index(atom.predicate)] = 1
        graph["globals"] = np.r_[atoms_globals, goal_globals, z]

        # Add nodes (one per object) and node features.
        graph["n_node"] = np.array(num_objects)
        node_features = np.zeros((num_objects, num_node_features))

        ## Add node features for obj types.
        for obj in state:
            obj_index = object_to_node[obj]
            type_index = self._node_feature_to_index[f"type_{obj.type.name}"]
            node_features[obj_index, type_index] = 1

        ## Add node features for unary atoms.
        for atom in atoms:
            if atom.predicate.arity != 1:
                continue
            obj_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[atom.predicate]
            node_features[obj_index, atom_index] = 1

        ## Add node features for unary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 1:
                continue
            obj_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[G(atom.predicate)]
            node_features[obj_index, atom_index] = 1

        ## Add node features for state.
        for obj in state:
            obj_index = object_to_node[obj]
            obj_type = obj.type.name
            for feat, val in zip(obj.type.feature_names, state[obj]):
                if CFG.exclude_domain_feat is not None:
                    if obj_type in CFG.exclude_domain_feat.keys():
                        if feat in CFG.exclude_domain_feat[obj_type]:
                            continue
                if 'pcd' in feat:
                    # pcd features
                    pcd = state[obj][obj.type.feature_names.index(feat)]
                    pcd = np.reshape(pcd, (-1, ))
                    for i, p in enumerate(pcd):
                        node_features[obj_index, self._node_feature_to_index[f"feat_{feat}_{i}"]] = p
                elif 'norm' in feat:
                    # pcd features
                    norm = state[obj][obj.type.feature_names.index(feat)]
                    norm = np.reshape(norm, (-1, ))
                    for i, n in enumerate(norm):
                        node_features[obj_index, self._node_feature_to_index[f"feat_{feat}_{i}"]] = n
                else:
                    feat_index = self._node_feature_to_index[f"feat_{feat}"]
                    node_features[obj_index, feat_index] = val

        graph["nodes"] = node_features

        # Deal with edge case (pun).
        num_edge_features = max(num_edge_features, 1)

        # Add edges (one between each pair of objects) and edge features.
        all_edge_features = np.zeros(
            (num_objects, num_objects, num_edge_features))

        ## Add edge features for binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[atom.predicate]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            all_edge_features[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[R(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[obj1_index, obj0_index, pred_index] = 1

        ## Add edge features for binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            all_edge_features[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(R(atom.predicate))]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[obj1_index, obj0_index, pred_index] = 1

        # Organize into expected representation.
        adjacency_mat = np.any(all_edge_features, axis=2)
        receivers, senders, edges = [], [], []
        for sender, receiver in np.argwhere(adjacency_mat):
            edge = all_edge_features[sender, receiver]
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)

        n_edge = len(edges)
        graph["edges"] = np.reshape(edges, [n_edge, num_edge_features])
        graph["receivers"] = np.reshape(receivers, [n_edge]).astype(np.int64)
        graph["senders"] = np.reshape(senders, [n_edge]).astype(np.int64)
        graph["n_edge"] = np.reshape(n_edge, [1]).astype(np.int64)

        return graph, object_to_node

    def _graphify_general_single_input(self, state: State, atoms: Set[GroundAtom],
                               goal: Set[GroundAtom]) -> Tuple[Dict, Dict]:
        all_objects = list(state)
        node_to_object = dict(enumerate(all_objects))
        object_to_node = {v: k for k, v in node_to_object.items()}
        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)
        
        sae_graph, _ = self._graphify_sae_single_input(state, atoms, goal)

        z = np.zeros(self._sae_model.U * self._sae_model.P * 2)
        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        graph = {}

        # Input globals: nullary predicates in atoms and goal.
        atoms_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        for atom in atoms:
            if atom.predicate.arity != 0:
                continue
            atoms_globals[self._nullary_predicates.index(atom.predicate)] = 1
        goal_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        for atom in goal:
            if atom.predicate.arity != 0:
                continue
            goal_globals[self._nullary_predicates.index(atom.predicate)] = 1
        graph["globals"] = np.r_[atoms_globals, goal_globals, z]

        # Add nodes (one per object) and node features.
        graph["n_node"] = np.array(num_objects)
        node_features = np.zeros((num_objects, num_node_features))

        ## Add node features for obj types.
        for obj in state:
            obj_index = object_to_node[obj]
            type_index = self._node_feature_to_index[f"type_{obj.type.name}"]
            node_features[obj_index, type_index] = 1

        ## Add node features for unary atoms.
        for atom in atoms:
            if atom.predicate.arity != 1:
                continue
            obj_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[atom.predicate]
            node_features[obj_index, atom_index] = 1

        ## Add node features for unary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 1:
                continue
            obj_index = object_to_node[atom.objects[0]]
            atom_index = self._node_feature_to_index[G(atom.predicate)]
            node_features[obj_index, atom_index] = 1

        ## Add node features for state.
        for obj in state:
            obj_index = object_to_node[obj]
            obj_type = obj.type.name
            for feat, val in zip(obj.type.feature_names, state[obj]):
                if CFG.exclude_domain_feat is not None:
                    if obj_type in CFG.exclude_domain_feat.keys():
                        if feat in CFG.exclude_domain_feat[obj_type]:
                            continue
                if 'pcd' in feat:
                    # pcd features
                    pcd = state[obj][obj.type.feature_names.index(feat)]
                    pcd = np.reshape(pcd, (-1, ))
                    for i, p in enumerate(pcd):
                        node_features[obj_index, self._node_feature_to_index[f"feat_{feat}_{i}"]] = p
                elif 'norm' in feat:
                    # pcd features
                    norm = state[obj][obj.type.feature_names.index(feat)]
                    norm = np.reshape(norm, (-1, ))
                    for i, n in enumerate(norm):
                        node_features[obj_index, self._node_feature_to_index[f"feat_{feat}_{i}"]] = n
                else:
                    feat_index = self._node_feature_to_index[f"feat_{feat}"]
                    node_features[obj_index, feat_index] = val

        graph["nodes"] = node_features

        # Deal with edge case (pun).
        num_edge_features = max(num_edge_features, 1)

        # Add edges (one between each pair of objects) and edge features.
        all_edge_features = np.zeros(
            (num_objects, num_objects, num_edge_features))

        ## Add edge features for binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[atom.predicate]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            all_edge_features[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms.
        for atom in atoms:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[R(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[obj1_index, obj0_index, pred_index] = 1

        ## Add edge features for binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(atom.predicate)]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            all_edge_features[obj0_index, obj1_index, pred_index] = 1

        ## Add edge features for reversed binary atoms in goal.
        for atom in goal:
            if atom.predicate.arity != 2:
                continue
            pred_index = self._edge_feature_to_index[G(R(atom.predicate))]
            obj0_index = object_to_node[atom.objects[0]]
            obj1_index = object_to_node[atom.objects[1]]
            # Note: the next line is reversed on purpose!
            all_edge_features[obj1_index, obj0_index, pred_index] = 1

        # Organize into expected representation.
        adjacency_mat = np.any(all_edge_features, axis=2)
        receivers, senders, edges = [], [], []
        for sender, receiver in np.argwhere(adjacency_mat):
            edge = all_edge_features[sender, receiver]
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)

        n_edge = len(edges)
        graph["edges"] = np.reshape(edges, [n_edge, num_edge_features])
        graph["receivers"] = np.reshape(receivers, [n_edge]).astype(np.int64)
        graph["senders"] = np.reshape(senders, [n_edge]).astype(np.int64)
        graph["n_edge"] = np.reshape(n_edge, [1]).astype(np.int64)

        return graph, sae_graph, object_to_node

    def learn_ama(
            self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom], _Option]]
    ) -> Tuple[List[GroundAtomTrajectory], Dict[Predicate, float]]:
        self._setup_output_specific_fields(data)
        # Set up the exemplar for the AMA model.
        example_input, example_object_to_node = self._graphify_ama_single_input(
            data[0][0], data[0][1], data[0][2])
        example_target = self._graphify_single_target(data[0][3],
                                                      example_input,
                                                      example_object_to_node)
        self._data_exemplar = (example_input, example_target)
        example_dataset = GraphDictDataset([example_input], [example_target])
        if CFG.gnn_use_pointnet:
            self._ama_gnn = setup_pointnet_graph_net(example_dataset,
                                        num_steps=self.model_cfg["ama"]["num_message_passing"],
                                        pointnet_feat_lst=CFG.gnn_pointnet_feat_lst,
                                        node_feature_to_index=self._node_feature_to_index.copy(),
                                        layer_size=self.model_cfg["ama"]["layer_size"],
                                        num_layers=self.model_cfg["ama"]["num_layers"],)
        else:
            self._ama_gnn = setup_graph_net(example_dataset,
                                        num_steps=self.model_cfg["ama"]["num_message_passing"],
                                        layer_size=self.model_cfg["ama"]["layer_size"],
                                        num_layers=self.model_cfg["ama"]["num_layers"],)
        if os.path.exists(CFG.fosae_ama_model_path):
            self._ama_gnn.load_state_dict(torch.load(CFG.fosae_ama_model_path))
            logging.info(f"Loaded AMA model from {CFG.fosae_ama_model_path}")
            return self._ama_gnn.state_dict()
        # Set up all the graphs, now using *all* the data.
        inputs = [(d[0], d[1], d[2]) for d in data]
        targets = [d[3] for d in data]
        graph_inputs = []
        graph_targets = []
        for (state, atoms, goal), target in tqdm(zip(inputs, targets)):
            graph_input, object_to_node = self._graphify_ama_single_input(
                state=state, atoms=atoms, goal=goal)
            graph_inputs.append(graph_input)
            graph_targets.append(
                self._graphify_single_target(target, graph_input,
                                             object_to_node))
        if CFG.fosae_ama_do_normalization:
            # Update normalization constants. Note that we do this for both
            # the input graph and the target graph.
            self._input_normalizers = compute_normalizers(graph_inputs, 
                                        normalize_nodes=self._node_is_rot, \
                                        normalize_edges=self._edge_is_rot, \
                                        normalize_globals=False)
            tgt_node_indicators = [False] * graph_targets[0]["nodes"].shape[1]
            tgt_edge_indicators = [False] * graph_targets[0]["edges"].shape[1]
            self._target_normalizers = compute_normalizers(graph_targets, 
                                        normalize_nodes=tgt_node_indicators, \
                                        normalize_edges=tgt_edge_indicators, \
                                        normalize_globals=False)
            graph_inputs = [
                normalize_graph(g, self._input_normalizers)
                for g in graph_inputs
            ]
            graph_targets = [
                normalize_graph(g, self._target_normalizers)
                for g in graph_targets
            ]
        # Run training.
        if CFG.fosae_use_validation_set:
            ## Split data, using 10% for validation.
            num_validation = max(1, int(len(inputs) * 0.1))
        else:
            num_validation = 0
        train_inputs = graph_inputs[num_validation:]
        train_targets = graph_targets[num_validation:]
        val_inputs = graph_inputs[:num_validation]
        val_targets = graph_targets[:num_validation]
        train_dataset = GraphDictDataset(train_inputs, train_targets)
        val_dataset = GraphDictDataset(val_inputs, val_targets)
        ## Set up Adam optimizer and dataloaders.
        training_bs = self.learning_cfg["ama"]["batch_size"]
        val_bs = self.learning_cfg["ama"]["batch_size_val"]
        lr = self.learning_cfg["ama"]["lr"]
        num_epochs = self.learning_cfg["ama"]["epochs"]
        optimizer = torch.optim.Adam(self._ama_gnn.parameters(),
                                     lr=lr)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=training_bs,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      persistent_workers=True,
                                      collate_fn=graph_batch_collate)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=val_bs,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=graph_batch_collate)
        dataloaders = {"train": train_dataloader, "val": val_dataloader}
        ## Launch training code.
        logging.info(f"Training GNN on {len(train_inputs)} examples using {CFG.device} device")
        best_model_dict = train_model(self._ama_gnn,
                                      dataloaders,
                                      optimizer=optimizer,
                                      criterion=self._criterion,
                                      global_criterion=self._global_criterion,
                                      num_epochs=num_epochs,
                                      do_validation=CFG.fosae_use_validation_set,
                                      device=CFG.device)
        return best_model_dict
        

    def _add_output_specific_fields_to_save_info(self, info: Dict) -> None:
        info["max_option_objects"] = self._max_option_objects

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        #  Step 1: Learning the State Auto Encoder model
        s = time.time()
        data = self.learn_sae(dataset)
        logging.info(f"SAE Training Done in {time.time()-s} seconds.")
        best_ama_model_dict = self.learn_ama(data)
        self._ama_gnn.load_state_dict(best_ama_model_dict)
        # save everything we get
        info = {
            "exemplar": self._data_exemplar,
            "ama_state_dict": self._ama_gnn.state_dict(),
            "sae_state_dict": self._sae_model.state_dict(),
            "nullary_predicates": self._nullary_predicates,
            "node_feature_to_index": self._node_feature_to_index,
            "sae_node_feature_to_index": self._sae_node_feature_to_index,
            "edge_feature_to_index": self._edge_feature_to_index,
            "input_normalizers": self._input_normalizers,
            "target_normalizers": self._target_normalizers,
        }
        self._add_output_specific_fields_to_save_info(info)
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_None.fosae", "wb") as f:
            pkl.dump(info, f)

    def _load_output_specific_fields_from_save_info(self, info: Dict) -> None:
        self._max_option_objects = info["max_option_objects"]
        self._sae_node_feature_to_index = info["sae_node_feature_to_index"]

    def load(self, online_learning_cycle: Optional[int]) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.fosae", "rb") as f:
            info = pkl.load(f)
        # Initialize fields from loaded dictionary.
        input_example, target_example = info["exemplar"]
        dataset = GraphDictDataset([input_example], [target_example])
        self._sae_model = setup_sae_net(
            CFG.fosae_max_n,
            len(info["sae_node_feature_to_index"]),
            self.model_cfg["sae"])
        self._sae_model.load_state_dict(info["sae_state_dict"])
        self._nullary_predicates = info["nullary_predicates"]
        self._node_feature_to_index = info["node_feature_to_index"]
        self._edge_feature_to_index = info["edge_feature_to_index"]
        self._input_normalizers = info["input_normalizers"]
        self._target_normalizers = info["target_normalizers"]
        if CFG.gnn_use_pointnet:
            self._ama_gnn = setup_pointnet_graph_net(dataset,
                                        num_steps=self.model_cfg["ama"]["num_message_passing"],
                                        pointnet_feat_lst=CFG.gnn_pointnet_feat_lst,
                                        node_feature_to_index=self._node_feature_to_index.copy(),
                                        layer_size=self.model_cfg["ama"]["layer_size"],
                                        num_layers=self.model_cfg["ama"]["num_layers"],)
        else:
            self._ama_gnn = setup_graph_net(dataset,
                                        num_steps=self.model_cfg["ama"]["num_message_passing"],
                                        layer_size=self.model_cfg["ama"]["layer_size"],
                                        num_layers=self.model_cfg["ama"]["num_layers"],)
        self._ama_gnn.load_state_dict(info["ama_state_dict"])
        self._load_output_specific_fields_from_save_info(info)

    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom], memory: Dict):
        del memory  # unused
        # Get output graph.
        in_graph_ama, in_graph_sae, object_to_node = self._graphify_general_single_input(
            state, atoms, goal)
        if CFG.fosae_ama_do_normalization:
            in_graph_ama = normalize_graph(in_graph_ama, self._input_normalizers)
        out_graph = get_single_model_prediction(self._ama_gnn, \
                                                self._sae_model, \
                                                in_graph_ama,
                                                in_graph_sae,
                                                CFG.device)
        if CFG.fosae_ama_do_normalization:
            out_graph = normalize_graph(out_graph,
                                        self._target_normalizers,
                                        invert=True)
        # Extract the output from the output graph.
        return self._extract_output_from_graph(out_graph, object_to_node)

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        assert self._ama_gnn is not None, "Learning hasn't happened yet!"
        assert self._sae_model is not None, "Learning hasn't happened yet!"
        return self._solve_with_shooting(task, timeout)

    def _solve_with_shooting(self, task: Task,
                             timeout: int) -> Callable[[State], Action]:
        # has simulator, shooting + checking until timeout or at least one try has
        # been done
        # we don't strictly use timeout since LatPlan is designed to be very slow...
        goal = task.goal
        memory: Dict = {}  # optionally updated by predict()
        start_time = time.perf_counter()
        # Keep trying until the timeout.
        tries: int = 0
        all_num_act: int = 0
        total_num_act: int = 0
        while (time.perf_counter() - start_time < timeout):
            # allow at least one try
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
                try:
                    possible_choices = self._predict(
                            state, atoms, task.goal, memory)
                except RuntimeError as e:
                    # FOSAE could fail in predictions, skip this try
                    logging.info(f"Runtime Error: {e}")
                    break
                # Sample an initiable option.
                for i in range(CFG.gnn_option_policy_shooting_max_samples):
                    selected_nsrt, objects, _ = possible_choices[i]
                    try:
                        ground_nsrt = selected_nsrt.ground(objects)
                    except AssertionError:
                        continue
                    if ground_nsrt.preconditions.issubset(atoms):
                        break
                else:
                    break  # out of the while loop for this shooting try
                cur_option = ground_nsrt.sample_option(state, goal=goal, rng=self._rng)
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
                if (time.perf_counter() - start_time > timeout) and \
                    (tries > 1):
                    break
                if num_act == 0:
                    break
        all_num_act += total_num_act
        logging.info(f"Shooting: {all_num_act} actions with {tries} tries")
        raise ApproachFailure("Failed to solve task in given time/try budget!",
                              info={
                                  "partial_refinements": (None, plan),
                              })
