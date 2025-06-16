"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar."""

from __future__ import annotations

import os
import copy
import glob
import time
import copy
import wandb
import yaml
import random
import itertools
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

from predicators import utils
from predicators.approaches.bilevel_learning_approach import BilevelLearningApproach
from predicators.settings import CFG
from predicators.predicate_search_score_functions import _BiSimBeliefScoreFunction
from predicators.structs import Dataset, GroundAtom, GroundAtomTrajectory, LowLevelTrajectoryReward, \
    _Option, ParameterizedOption, DummyPredicate, NeuralPredicate, Predicate, Object, State, Task, Type, \
    Action
from predicators.structs import NSRT, PNAD, GroundAtomTrajectory, \
    LowLevelTrajectory, ParameterizedOption, Predicate, Segment, Task

_Output = TypeVar("_Output")  # a generic type for the output of this GNN

################################################################################
#                                 Approach                                     #
################################################################################


class BiSimApproach(BilevelLearningApproach):
    """An approach that invents predicates by learn a GNN that maps continous Graph 
    to discrete space. Using Action Effect Theorem. """

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        assert os.path.exists(CFG.neupi_load_pretrained), "Should have the pretrained neural predicates provided"
        assert not CFG.load_neupi_from_json, "Should not load from json"

    @classmethod
    def get_name(cls) -> str:
        return "bisim"
    
    def _select_predicates_by_score_search(
            self, huge_ae_matrixe: np.ndarray,
            all_predicates_info: Dict[int, Tuple[Predicate, List[int], bool]],
            score_function: _BiSimBeliefScoreFunction) \
            -> Tuple[Set[Predicate], np.ndarray, List[Predicate]]:
        """Perform a greedy search over predicate sets."""

        # Start the search with no candidates.
        provided_effect_predicates = []
        provided_prec_predicates = []
        basic_matrix = []
        basic_pred_ent_idx = []
        possible_cols = []
        possible_predicates = []
        possible_pred_entidx = []
        for colind in all_predicates_info.keys():
            if all_predicates_info[colind][2]:
                logging.info(f"Provided Predicate: {all_predicates_info[colind][0]}")
                # this is provided, has to be in the final set
                provided_effect_predicates.append(all_predicates_info[colind][0])
                basic_matrix.append(huge_ae_matrixe[:, colind])
                basic_pred_ent_idx.append(all_predicates_info[colind][1])
            else:
                logging.info(f"Possible Predicate: {all_predicates_info[colind][0]}")
                possible_cols.append(colind)
                possible_predicates.append(all_predicates_info[colind][0])
                possible_pred_entidx.append(all_predicates_info[colind][1])
        # We might also want the pre-condition only predicates in the score computation
        for pred in list(self._initial_predicates):
            if pred in provided_effect_predicates:
                continue
            provided_prec_predicates.append(pred)
        # start hill-climbing + matrix search
        # First Try Vanilla Hill-Climbing Search
        searched_predicate_comp = []
        final_predicates = copy.deepcopy(provided_effect_predicates)
        level = 0
        go_on = True
        last_matrix = torch.stack(basic_matrix, dim=1)
        last_ent_idx = copy.deepcopy(basic_pred_ent_idx)
        search_record = {}
        best_score = float("inf")
        s = time.time()
        while go_on:
            logging.info(f"Search Level {level}...")
            add_idx = -1
            for idx, p_pred in enumerate(possible_predicates):
                if p_pred in final_predicates:
                    continue
                # note col id != idx, add matrix comulmn need to use col id
                col_idx = possible_cols[idx]
                try_matrix = torch.cat([last_matrix, huge_ae_matrixe[:, col_idx:col_idx+1]], dim=1)
                try_pred_ent_idx = copy.deepcopy(last_ent_idx)
                try_pred_ent_idx.append(possible_pred_entidx[idx])
                if (not self.check_matrix_consistency(try_matrix, try_pred_ent_idx)) and \
                    (self.check_matrix_consistency(last_matrix, last_ent_idx)):
                    #  last matrix sat consistency but curr not sat, skip it
                    logging.info(f"Matrix is not consistent, skip...")
                    continue
                try_predicates = []
                for pred in final_predicates:
                    try_predicates.append(pred)
                try_predicates.append(p_pred)
                searched_predicate_comp.append(set(try_predicates))
                s = time.time()
                # also consider the precondition only predicates
                try_predicates.extend(provided_prec_predicates)
                score = score_function.evaluate(try_matrix, try_predicates, try_pred_ent_idx, CFG.strips_learner)
                logging.info(f"Predicate set {try_predicates} has score {score} in {time.time()-s} seconds.")
                if score < best_score:
                    # solve at least one more task
                    logging.info(f"New smallest score: {score}")
                    best_score = score
                    add_idx = idx
            if add_idx != -1:
                logging.info(f"New best score at level {level}: {best_score}")
                logging.info(f"Adding Predicate {possible_predicates[add_idx]}...")
                final_predicates.append(possible_predicates[add_idx])
                # note col id != idx, add matrix comulmn need to use col id
                col_idx_add = possible_cols[add_idx]
                last_matrix = torch.cat([last_matrix, huge_ae_matrixe[:, col_idx_add:col_idx_add+1]], dim=1)
                last_ent_idx.append(possible_pred_entidx[add_idx])
                search_record[level] = (possible_predicates[add_idx], best_score)
            else:
                go_on = False
                logging.info(f"No more predicates to add at level {level} Stop searching.")
        return set(final_predicates), last_matrix, final_predicates, last_ent_idx

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        #  Step 1: Initializing individual NN predicates by Bi-Level Learning
        s = time.time()
        learning_trajectories, init_atom_traj = self.learn_neural_predicates(dataset)
        logging.info(f"Neural Predicate Invention Done in {time.time()-s} seconds.")
        if CFG.neupi_save_init_atom_dataset:
            save_path = utils.get_approach_save_path_str()
            with open(f"{save_path}_.init_atom_traj", "wb") as f:
                pkl.dump(init_atom_traj, f)
        # Purne Equivalence Classes
        # Step 2: Get all the columns into one big matrix and their info, including provided
        num_traj = len(learning_trajectories)
        traj4equavalence = [learning_trajectories[i] for i in range(int(num_traj * CFG.neupi_equ_dataset))]
        huge_matrix, colind2info = self._get_invented_pruned_predicates(traj4equavalence)
        all_pred = set(info[0] for info in colind2info.values())
        new_predicate = set()
        for pred in list(all_pred):
            if pred in self._initial_predicates:
                continue
            new_predicate.add(pred)
        # Step 3: Compute the AAAI Objective for each "possible" matrix/predicate set
        # 1. construct atom dataset for all the predicates
        traj4search = [init_atom_traj[i] for i in range(int(num_traj * CFG.neupi_pred_search_dataset))]
        s = time.time()
        atom_dataset_part = utils.add_ground_atom_dataset(
                        traj4search,
                        new_predicate)
        # remeber to add the initial predicates, which may not in the effect matrix (unchanged)
        all_pred = all_pred | self._initial_predicates
        # 2. Score Function Def
        # Note that need to use full _train_tasks
        score_function = _BiSimBeliefScoreFunction(
                self._initial_predicates,
                atom_dataset_part, self._train_tasks, self.ae_row_names,
                CFG.neupi_aaai_metric)
        self._learned_predicates, self.ae_matrix_tgt, self.ae_col_names_all, self.ae_col_ent_idx = \
                self._select_predicates_by_score_search(
                huge_matrix, colind2info, score_function)
        logging.info(f"Predicate Selection Done in {time.time()-s} seconds.")
        logging.info("Final AE matrix (Add): {}".format(self.ae_matrix_tgt[:, :, 0]))
        if self.ae_matrix_tgt.shape[-1] == 2:
            logging.info("Final AE matrix (Del): {}".format(self.ae_matrix_tgt[:, :, 1]))
        logging.info("Final (Ordered) Effect Predicates: {}".format(self.ae_col_names_all))
        # For efficiency, just prune it, no need to ground again
        # Note, remeber to keep the precondition only predicates in initial predicates
        if CFG.neupi_pred_search_dataset == 1:
            # only needs to prune the atom dataset
            atom_dataset = utils.prune_ground_atom_dataset(
                atom_dataset_part,
                self._learned_predicates | self._initial_predicates)
        else:
            # needs to ground again from the beginning
            atom_dataset = utils.add_ground_atom_dataset(
                init_atom_traj,
                self._learned_predicates)
        logging.info("Learning NSRT from the learned predicates.")
        # Finally, learn NSRTs via superclass, using all the kept predicates.
        annotations = None
        if dataset.has_annotations:
            annotations = dataset.annotations
        logging.info("Learning NSRTs from the provided atom dataset.")
        self._learn_nsrts(learning_trajectories,
                          atom_dataset,
                          annotations)
        # save everything we get
        save_path = utils.get_approach_save_path_str()
        saved_info = {"learned_ae_pred_info": copy.deepcopy(self.learned_ae_pred_info)
                      }
        saved_info['node_feature_to_index'] = self._node_feature_to_index
        saved_info['edge_feature_to_index'] = self._edge_feature_to_index
        saved_info['node_is_rot'] = self._node_is_rot
        saved_info['edge_is_rot'] = self._edge_is_rot
        saved_info['input_normalizers'] = self._input_normalizers
        # we will save the selected pred info instead of the predicate, since it is neural-related classifier
        selected_pred = {}
        selected_pred2dummy = {}
        for pred in self._learned_predicates:
            for col, info in colind2info.items():
                if info[0] == pred:
                    # this predicate is a template
                    dummy_pred = info[-1][0]
                    name = pred.name
                    types = pred.types
                    # this predicate is specifically for this one
                    new_dummy_pred = DummyPredicate(name=name, types=types)
                    selected_pred2dummy[pred] = new_dummy_pred
                    # importanty, each dummy pred could have multiple real preds
                    if dummy_pred in selected_pred:
                        selected_pred[dummy_pred].append((new_dummy_pred, info[-1][1], info[-1][2]))
                    else:
                        selected_pred[dummy_pred] = [(new_dummy_pred, info[-1][1], info[-1][2])]
                    break
        saved_info['selected_pred'] = selected_pred
        # we will save the "dummy" nsrts with the neural predicates replaced by dummy ones with the same name
        dummy_nsrts = utils.replace_nsrt_predicates(self._nsrts, selected_pred2dummy)
        saved_info['dummy_nsrts'] = dummy_nsrts
        # finally, save the info
        with open(f"{save_path}.bisim_info", "wb") as f:
            pkl.dump(saved_info, f)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        self.load_from_pkl()