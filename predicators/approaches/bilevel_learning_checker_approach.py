"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar."""

from __future__ import annotations

import os
import wandb
import json
import functools
import logging
from z3 import *
from tqdm import tqdm
from gym.spaces import Box
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set, Tuple, TypeVar

import numpy as np
import dill as pkl
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader

from predicators import utils
from predicators.approaches.bilevel_learning_approach import BilevelLearningApproach
from predicators.gnn.neupi import DummyModelIndex, setup_neupi_optimizer
from predicators.gnn.neupi_utils import train_init_dummy_model
from predicators.gnn.gnn_utils import GraphTransC2DDataset, action_graph_batch_collate
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.settings import CFG
from predicators.structs import MDPDataset, GroundAtom, GroundAtomTrajectory, LowLevelTrajectoryReward, \
    _Option, ParameterizedOption, DummyPredicate, NeuralPredicate, Predicate, Object, State, Task, Type

_Output = TypeVar("_Output")  # a generic type for the output of this GNN

################################################################################
#                          Programmatic classifiers                            #
################################################################################



################################################################################
#                                 Approach                                     #
################################################################################


class ivntrCheckerApproach(BilevelLearningApproach):
    """An approach that invents predicates by learn a GNN that maps continous Graph 
    to discrete space. Using Action Effect Theorem. """

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self.debug = True
        self.check_first = True
        self.ae_col_belief = [] # checker do not need belief
        self.ae_col_names_all = []
        self.ae_learned_columns = []
        self.ae_provided_columns = []
        self.ae_pred_nn = []
        logging.info("Initialize Predicate Columns:")
        # provided predicates
        # this is used to generate the ae matrix
        for pred in sorted(self._initial_predicates):
            if pred.name in CFG.neupi_non_effect_predicates:
                # We don't want to invent predicates that are not effect predicates
                continue
            if pred.name in ['CameraReadingTaken', 'GeigerReadingTaken', 'InfraredReadingTaken']:
                self.ae_col_names_all.append(pred)
                # needs to distinguish provided and learned predicates
                self.ae_learned_columns.append(1)
                self.ae_provided_columns.append(1)
                self.ae_pred_nn.append(None)
                logging.info("Provided Predicate (No Learning): {}".format(pred.name))
            else:
                self.ae_col_names_all.append(pred)
                self.ae_learned_columns.append(0)
                self.ae_provided_columns.append(0)
                self.ae_pred_nn.append(None)
                logging.info("Provided Predicate (Learning): {}".format(pred.name))

    @classmethod
    def get_name(cls) -> str:
        return "ivntr_checker"

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates
    
    def initialize_ae_matrix(self, data: List[Tuple[State, Set[GroundAtom], State, Set[GroundAtom], \
                                   _Option, str]], channels: int=1, rules: List=[]) -> torch.Tensor:
        """
        Generate the initial binary action effect matrix.
        """
        width = len(self.ae_col_names_all)
        height = len(self.ae_row_names)

        assert channels <= 2, "Only support at most 2 channels"

        # Construct SMT solver to find a matrix that satisfies the constraints
        solver = Solver()
        local_entities = [Bool(f'x_{row}_{col}_{channel}') for row in range(height) \
                          for col in range(width) for channel in range(channels)]

        # Non-zero constraints for rows and columns
        for row in range(height):
            exist_non_zero = Or([local_entities[row * width * channels + col * channels + channel] \
                                 for col in range(width) for channel in range(channels)])
            solver.add(exist_non_zero)

        for col in range(width):
            exist_non_zero = Or([local_entities[row * width * channels + col * channels + channel] \
                                 for row in range(height) for channel in range(channels)])
            solver.add(exist_non_zero)

        # Across all the chanels of each entry, there exist at most one 1
        for row in range(height):
            for col in range(width):
                variables = [local_entities[row * width * channels + col * channels + channel] for channel in range(channels)]
                for i in range(len(variables)):
                    for j in range(i + 1, len(variables)):
                        # Add the constraint that at most one of the two variables can be true
                        solver.add(Or(Not(variables[i]), Not(variables[j])))

        # Ensure any two pairs of rows are different in at least one column and one channel
        for row1 in range(height):
            for row2 in range(row1 + 1, height):
                diff_constraints = []
                for col in range(width):
                    for channel in range(channels):
                        diff_constraints.append(local_entities[row1 * width * channels + col * channels + channel] \
                                                != local_entities[row2 * width * channels + col * channels + channel])
                solver.add(Or(diff_constraints))

        # Read the data and pose "zero" constraints by comparing the continous states
        # Distill intial constraints from transition data
        all_zero_ae_matrix = torch.zeros((height, width, channels), dtype=int)
        for state, _, state_, _, action, _ in data:
            action_index = self.ae_row_names.index(action.parent)
            operated_objs = action.objects
            for affected_obj in operated_objs:
                s = state[affected_obj]
                s_ = state_[affected_obj]
                diff = s_ - s
                if not np.all(diff == 0):
                    for pred in self.ae_col_names_all:
                        arity = pred.arity
                        col_index = self.ae_col_names_all.index(pred)
                        if arity == 1:
                            types = pred.types
                            if types[0] == affected_obj.type:
                                all_zero_ae_matrix[action_index, col_index, :] = 1
                        elif arity == 2:
                            types = pred.types
                            if types[0] == affected_obj.type or types[1] == affected_obj.type:
                                all_zero_ae_matrix[action_index, col_index, :] = 1
        constraints = []
        for row in range(height):
            if all_zero_ae_matrix[row].sum() == 0:
                # this action never appears
                continue
            for col in range(width):
                for channel in range(channels):
                    if all_zero_ae_matrix[row, col, channel] == 0:
                        rule = ('position', row, col, channel, False)
                        constraints.append(rule)
                        solver.add(local_entities[row * width * channels + col * channels + channel] == False)

        # Read the data and pose constraints for the given predicates (we can know it from data)
        all_ae_matrix_dummy = torch.zeros((height, width, channels), dtype=int) - 1
        for pred in self.ae_col_names_all:
            pred_index = self.ae_col_names_all.index(pred)
            all_ae_matrix_dummy[:, pred_index, :] = 0 # assmues no effect

        for _, atom, _, atom_, action, _ in data:
            action_index = self.ae_row_names.index(action.parent)
            for at in atom_:
                if at.predicate.arity not in [1, 2]:
                    continue
                if at.predicate not in self.ae_col_names_all:
                    continue
                if at not in atom:
                    # this is an add effect
                    pred_index = self.ae_col_names_all.index(at.predicate)
                    assert all_ae_matrix_dummy[action_index, pred_index, 0] >= 0, "Should be zero"
                    all_ae_matrix_dummy[action_index, pred_index, 0] = 1

            if channels == 2:
                for at in atom:
                    if at.predicate.arity not in [1, 2]:
                        continue
                    if at.predicate not in self.ae_col_names_all:
                        continue
                    if at not in atom_:
                        # this is a delete effect
                        pred_index = self.ae_col_names_all.index(at.predicate)
                        assert all_ae_matrix_dummy[action_index, pred_index, 0] >= 0, "Should be zero"
                        all_ae_matrix_dummy[action_index, pred_index, 1] = 1

        for col in range(width):
            if all_ae_matrix_dummy[:, col].sum() == 0:
                # this effect never appears
                raise ValueError("This effect predicate never appears in data, bug!")
            if all_ae_matrix_dummy[:, col].sum() == -1 * height * channels:
                # this effect predicate is not provided
                continue
            for row in range(height):
                if all_ae_matrix_dummy[row, :, :].sum() == 0:
                    # this action never appears
                    continue
                for channel in range(channels):
                    if all_ae_matrix_dummy[row, col, channel] == 0:
                        rule = ('position', row, col, channel, False)
                        if rule not in constraints:
                            constraints.append(rule)
                            solver.add(local_entities[row * width * channels + col * channels + channel] == False)
                    elif all_ae_matrix_dummy[row, col, channel] == 1:
                        rule = ('position', row, col, channel, True)
                        if rule not in constraints:
                            constraints.append(rule)
                            solver.add(local_entities[row * width * channels + col * channels + channel] == True)
                    else:
                        raise ValueError("Should be 0 or 1, got {}".format(all_ae_matrix_dummy[row, col, channel]))
                    
        if solver.check() == sat:
            model = solver.model()
            matrix = torch.zeros((height, width, channels), dtype=int)
            for row in range(height):
                for col in range(width):
                    for channel in range(channels):
                        if is_true(model.evaluate(local_entities[row * width * channels + col * channels + channel])):
                            matrix[row, col, channel] = 1
            return matrix, constraints
        else:
            raise ValueError("No satisfying initial matrix found")

    def _setup_input_fields(
        self, data: List[Tuple[State, Set[GroundAtom], State, Set[GroundAtom], \
                                 _Option, str]]
    ) -> None:
        """
        For checking purpose, the input is the grounded binary graph.
        Graph input only has continous object features.
        """
        nullary_predicates_set = set()
        unary_predicates_set = set()
        binary_predicates_set = set()

        # Go through the data, identifying the types/attributes.
        for _, atoms, _, atoms_, _, _ in data:
            for atom in atoms:
                arity = atom.predicate.arity
                assert arity <= 2, "Predicates with arity > 2 are not supported"
                if atom.predicate.name in CFG.neupi_non_effect_predicates:
                    # We don't want to invent predicates that are not effect predicates
                    continue 
                elif arity == 1:
                    unary_predicates_set.add(atom.predicate)
                elif arity == 2:
                    binary_predicates_set.add(atom.predicate)
            for atom in atoms_:
                arity = atom.predicate.arity
                assert arity <= 2, "Predicates with arity > 2 are not supported"
                if atom.predicate.name in CFG.neupi_non_effect_predicates:
                    # We don't want to invent predicates that are not effect predicates
                    continue 
                elif arity == 1:
                    unary_predicates_set.add(atom.predicate)
                elif arity == 2:
                    binary_predicates_set.add(atom.predicate)

        unary_predicates = sorted(unary_predicates_set)
        binary_predicates = sorted(binary_predicates_set)
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        # Initialize input node features.
        self._node_feature_to_index = {}
        index = 0
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[unary_predicate] = index
            index += 1

        # Initialize input edge features. Concat of two node features.
        self._edge_feature_to_index = {}
        index = 0
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[binary_predicate] = index
            index += 1
            type0, type1 = binary_predicate.types
            if type0 == type1:
                # we need to distinguish var order if this predicates accept the same 2 types
                self._edge_feature_to_index[R(binary_predicate)] = index
                index += 1

    def _setup_output_fields(
        self, data: List[Tuple[State, Set[GroundAtom], State, Set[GroundAtom], \
                                 _Option, str]]
    ) -> None:
        invented_unary_predicates_set = set()
        invented_binary_predicates_set = set()

        # Go through the data, identifying the types/attributes.
        for _, atoms, _, atoms_, _, _ in data:
            for atom in atoms:
                arity = atom.predicate.arity
                assert arity <= 2, "Predicates with arity > 2 are not supported"
                if atom.predicate.name in CFG.neupi_non_effect_predicates:
                    # We don't want to invent predicates that are not effect predicates
                    continue 
                elif arity == 1:
                    invented_unary_predicates_set.add(atom.predicate)
                elif arity == 2:
                    invented_binary_predicates_set.add(atom.predicate)

            for atom in atoms_:
                arity = atom.predicate.arity
                assert arity <= 2, "Predicates with arity > 2 are not supported"
                if atom.predicate.name in CFG.neupi_non_effect_predicates:
                    # We don't want to invent predicates that are not effect predicates
                    continue 
                elif arity == 1:
                    invented_unary_predicates_set.add(atom.predicate)
                elif arity == 2:
                    invented_binary_predicates_set.add(atom.predicate)

        invented_unary_predicates_set = sorted(invented_unary_predicates_set)
        invented_binary_predicates_set = sorted(invented_binary_predicates_set)

        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        # Initialize output node features.
        index_invent = 0
        for unary_predicate in invented_unary_predicates_set:
            self._invent_u_predicate_to_index[unary_predicate] = index_invent
            index_invent += 1
        
        # Initialize input edge features.
        index_invent = 0
        for binary_predicate in invented_binary_predicates_set:
            self._invent_b_predicate_to_index[binary_predicate] = index_invent
            index_invent += 1
            type0, type1 = binary_predicate.types
            if type0 == type1:
                # we need to distinguish var order if this predicates accept the same 2 types
                self._invent_b_predicate_to_index[R(binary_predicate)] = index_invent
                index_invent += 1

    def _graphify_single_input(self, state: State, atoms: Set[GroundAtom]) -> Tuple[Dict, Dict]:
        all_objects = list(state)
        node_to_object = dict(enumerate(all_objects))
        object_to_node = {v: k for k, v in node_to_object.items()}
        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)

        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        graph = {}

        # Input globals: nullary predicates in atoms and goal.
        # atoms_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        # for atom in atoms:
        #     if atom.predicate.arity != 0:
        #         continue
        #     atoms_globals[self._nullary_predicates.index(atom.predicate)] = 1
        # goal_globals = np.zeros(len(self._nullary_predicates), dtype=np.int64)
        # graph["globals"] = np.r_[atoms_globals, goal_globals]
        # Add nodes (one per object) and node features.
        graph["n_node"] = np.array(num_objects)
        node_features = np.zeros((num_objects, num_node_features))
        if self.check_first:
            predicates = self._get_current_predicates()
            parsed_atoms = utils.abstract(state, predicates)
            ## Add node features for unary atoms.
            for atom in parsed_atoms:
                if atom.predicate.arity != 1:
                    continue
                if atom.predicate not in self._node_feature_to_index:
                    continue
                obj_index = object_to_node[atom.objects[0]]
                atom_index = self._node_feature_to_index[atom.predicate]
                node_features[obj_index, atom_index] = 1
        else:
            ## Add node features for unary atoms.
            for atom in atoms:
                if atom.predicate.arity != 1:
                    continue
                if atom.predicate not in self._node_feature_to_index:
                    continue
                obj_index = object_to_node[atom.objects[0]]
                atom_index = self._node_feature_to_index[atom.predicate]
                node_features[obj_index, atom_index] = 1

        graph["nodes"] = node_features

        # Deal with edge case (pun).
        num_edge_features = max(num_edge_features, 1)

        # Add edges (one between each pair of objects) and edge features.
        all_edge_features = np.zeros(
            (num_objects, num_objects, num_edge_features))

        ## Add edge features
        if self.check_first:
            ## Add edge features
            for atom in parsed_atoms:
                if atom.predicate.arity != 2:
                    continue
                if not atom.predicate in self._edge_feature_to_index:
                    continue
                pred_index = self._edge_feature_to_index[atom.predicate]
                obj0_index = object_to_node[atom.objects[0]]
                obj1_index = object_to_node[atom.objects[1]]
                all_edge_features[obj0_index, obj1_index, pred_index] = 1

            ## Add edge features for reversed binary atoms.
            for atom in parsed_atoms:
                if atom.predicate.arity != 2:
                    continue
                if not R(atom.predicate) in self._edge_feature_to_index:
                    continue
                pred_index = self._edge_feature_to_index[R(atom.predicate)]
                obj0_index = object_to_node[atom.objects[0]]
                obj1_index = object_to_node[atom.objects[1]]
                # Note: the next line is reversed on purpose!
                all_edge_features[obj1_index, obj0_index, pred_index] = 1
        else:
            for atom in atoms:
                if atom.predicate.arity != 2:
                    continue
                if not atom.predicate in self._edge_feature_to_index:
                    continue
                pred_index = self._edge_feature_to_index[atom.predicate]
                obj0_index = object_to_node[atom.objects[0]]
                obj1_index = object_to_node[atom.objects[1]]
                all_edge_features[obj0_index, obj1_index, pred_index] = 1

            ## Add edge features for reversed binary atoms.
            for atom in atoms:
                if atom.predicate.arity != 2:
                    continue
                if not R(atom.predicate) in self._edge_feature_to_index:
                    continue
                pred_index = self._edge_feature_to_index[R(atom.predicate)]
                obj0_index = object_to_node[atom.objects[0]]
                obj1_index = object_to_node[atom.objects[1]]
                # Note: the next line is reversed on purpose!
                all_edge_features[obj1_index, obj0_index, pred_index] = 1

        # Organize into expected representation.
        # this should be an all-connected graph
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

        # Add global features.
        graph["globals"] = None

        return graph, object_to_node

    def gen_graph_data(self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom], _Option]],
                          batch_size: int,
                          current_pred: Predicate,
                          ae_matrix: torch.Tensor) -> List[Dict]:
        all_data = []
        logging.info("Generating Graph Data with Current AE matrix...")
        for state, atoms, state_, atoms_, action, action_label in tqdm(data):
            # state s
            input_graph, object_to_node = self._graphify_single_input(state, atoms)
            # partitioned discrete graph (invented predicates)
            target_graph, action_info = self._graphify_single_output(
                action, action_label, current_pred, ae_matrix, 0, input_graph, object_to_node)
            # state s'
            input_graph_, object_to_node_ = self._graphify_single_input(state_, atoms_)
            assert object_to_node_ == object_to_node
            # partitioned discrete graph (invented predicates)
            target_graph_, _ = self._graphify_single_output(
                action, action_label, current_pred, ae_matrix, 1, input_graph_, object_to_node)
            data_dict = {
                'input': input_graph,
                'target': target_graph,
                'input_': input_graph_,
                'target_': target_graph_,
                'action_info': action_info
            }
            all_data.append(data_dict)

        train_dataset = GraphTransC2DDataset(all_data[:int(len(all_data)*0.8)])
        val_dataset = GraphTransC2DDataset(all_data[int(len(all_data)*0.8):])

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=2,
                                      collate_fn=action_graph_batch_collate)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    collate_fn=action_graph_batch_collate)

        return train_dataloader, val_dataloader

    def learn_neural_predicates(
        self, dataset: MDPDataset
    ) -> Tuple[List[GroundAtomTrajectory], Dict[Predicate, float]]:
        """Learn the Neural predicates by Action Effect Martix Identification."""
        logging.info("Constructing NeuPi Data...")
        # 1. Generate data from the dataset. This is general
        data, trajectories = self._generate_data_from_dataset(dataset)
        # 2. Setup the input fields for the neural predicate, this is general
        self._setup_input_fields(data)
        # 3. Setup the output fields for the neural predicate, this is general (just mask the superv signal)
        self._setup_output_fields(data)
        # 4. Get the initial marix and constraints (for all the predicates)
        initial_ae_matrix, initial_c = self.initialize_ae_matrix(data, CFG.neupi_ae_matrix_channel)
        self.ae_constraints.extend(initial_c)

        # 4. Start Learning the columns one by one
        for pred_idx, curr_pred in enumerate(self.ae_col_names_all):
            if self.ae_learned_columns[pred_idx]:
                # this is learned/provided, skip
                continue
            logging.info(f"*****Checking Predicate: {curr_pred}*****")
            if curr_pred.arity == 1:
                indx = self._invent_u_predicate_to_index[curr_pred]
            elif curr_pred.arity == 2:
                indx = self._invent_b_predicate_to_index[curr_pred]
            predicate_neural_model = DummyModelIndex(
                random=True,
                arity=curr_pred.arity,
                idx=indx
            )
            optimizer_config = {
                'type': "Adam",
                'kwargs': {
                    'lr': 0.001,
                }
            }
            optimizer, scheduler = setup_neupi_optimizer(predicate_neural_model, \
                                                        optimizer_config)

            # Step2: Start Iteration
            logging.info(f"Checking This Predicate...")
            logging.info("Current target ae matrix (Add): {}".format(initial_ae_matrix[:, :, 0]))
            if initial_ae_matrix.shape[2] == 2:
                logging.info("Current target ae matrix (Del): {}".format(initial_ae_matrix[:, :, 1]))
            # Inside Step 1: Gen data use current ae matrix
            logging.info("Generating data using current ae matrix...")
            train_dataloader, val_dataloader = self.gen_graph_data(data, 512, curr_pred, initial_ae_matrix)
            # Inside Step 2: Use current label to train the model
            best_model_dict_iter = train_init_dummy_model(
                                        predicate_neural_model,
                                        train_dataloader,
                                        val_dataloader,
                                        optimizer=optimizer,
                                        criterion=self._supervise_criterion,
                                        num_epochs=3,
                                        gumbel_temperature=0.66,
                                        val_freq=1,
                                        device=CFG.device)

        # save the found best ae matrix and corresponding model
        logging.info(f"Checker finished")
        return 

    def learn_from_offline_dataset(self, dataset: MDPDataset) -> None:
        #  Step 1: Initializing GNN predicates by Action Effect Theorem
        self.learn_neural_predicates(dataset)

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectoryReward],
                     atom_dataset: Optional[List[GroundAtomTrajectory]],
                     re_parse: bool,
                     annotations: Optional[List[Any]],
                     dataset_fname: Optional[str]='') -> None:
        if re_parse:
            logging.info("Re-parsing the data.")
            # Apply predicates to data, producing a dataset of abstract states.
            ground_atom_dataset = utils.create_ground_atom_dataset(
                trajectories, self._get_current_predicates())
            utils.save_ground_atom_dataset(ground_atom_dataset, dataset_fname)
        else:
            logging.info("Using the provided ground atom dataset.")
            ground_atom_dataset = atom_dataset

        # Learn NSRTs from the ground atom dataset, with belief assumptions.
        operator_belief = {
            'row_names': self.ae_row_names,
            'col_names': self.ae_col_names,
            'ae_matrix': self.ae_matrix_tgt.cpu().numpy()
        }

        self._nsrts, self._segmented_trajs, self._seg_to_nsrt = \
            learn_nsrts_from_data(trajectories,
                                  self._train_tasks,
                                  self._get_current_predicates(),
                                  self._initial_options,
                                  self._action_space,
                                  ground_atom_dataset,
                                  sampler_learner=CFG.sampler_learner,
                                  annotations=annotations,
                                  operator_belief=operator_belief)
        # save_path = utils.get_approach_save_path_str()
        # with open(f"{save_path}_.NSRTs", "wb") as f:
        #     pkl.dump(list(self._nsrts), f)