"""
An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar. Additionally, we apply the "consist" and
"informative" condition for each independent predicate and do a filter before they
are considered for the final set of predicates.
"""

from __future__ import annotations

import os
import yaml
import abc
import itertools
import logging
from dataclasses import dataclass, field
from functools import cached_property
from operator import le
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set, Tuple

import torch
from tqdm import tqdm
import dill as pkl
import time as t
from gym.spaces import Box
from z3 import *
from sklearn.mixture import GaussianMixture as GMM

from predicators import utils
from predicators.gnn.neupi_utils import mdp2positive, extract_ae_vector, compute_guidance_vector, \
    select_columns
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.approaches.grammar_search_invention_approach import GrammarSearchInventionApproach
from predicators.approaches.bilevel_learning_approach import BilevelLearningApproach
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.predicate_search_score_functions import \
    _PredicateSearchScoreFunction, create_score_function
from predicators.settings import CFG
from predicators.predicate_search_score_functions import _OperatorBeliefScoreFunction
from predicators.structs import DummyPredicate, Dataset, GroundAtomTrajectory, GroundAtom, \
    LowLevelTrajectory, ParameterizedOption, Predicate, Segment, State, Task, Type, _Option, \
    MDPDataset


################################################################################
#                                 Approach                                     #
################################################################################


class GrammarSearchConditionApproach(GrammarSearchInventionApproach, \
                                     BilevelLearningApproach):
    """An approach that invents predicates by searching over candidate sets,
    with the candidates proposed from a grammar. Before search, we use the
    "consist" and "informative" condition for each independent predicate and
    do a filter before they are considered for the final set of predicates.
    Notice that the inherited class resolution order.
    """

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        for pred in self.learned_ae_pred_info.keys():
            if self.learned_ae_pred_info[pred]['provided']:
                continue
            self.learned_ae_pred_info[pred]['all_candidates'] = []
            self.learned_ae_pred_info[pred]['all_cost'] = []

    @classmethod
    def get_name(cls) -> str:
        return "grammar_search_condition"

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates
    
    def check_sat_vec(self, pred: DummyPredicate, \
                    ae_vec_flipped: torch.Tensor) \
                    -> List[torch.Tensor]:
        """
        Generate the action effect matrix guided by entropy.
        """
        constraints = self.learned_ae_pred_info[pred]['constraints']
        # it is already clamped, do not normalize with max
        # n_row x (n_channel + 1)
        width = 1
        height = len(self.ae_row_names)
        channels = CFG.neupi_ae_matrix_channel
        # check if the sampled matrix is satisfying all constraints
        solver = Solver()
        local_entities = [Bool(f'x_{row}_{col}_{channel}') for row in range(height) \
                        for col in range(width) for channel in range(channels)]
        
        solver = self.add_general_col_constraints(solver, local_entities, width, height, channels)
        constraints = self.learned_ae_pred_info[pred]['constraints']
        for rule in constraints:
            c_type = rule[0]
            if c_type == 'position':
                row, col, channel, value = rule[1], rule[2], rule[3], rule[4]
                assert col == 0, "Should be 0 for a vector"
                solver.add(local_entities[row * width * channels + col * channels + channel] == value)
            else:
                raise ValueError('Unknown constraint type')
            
        # add all the entry in flipped matrix
        for row in range(height):
            for channel in range(channels):
                if ae_vec_flipped[row, channel] == 1:
                    solver.add(local_entities[row * width * channels + 0 * channels + channel] == True)
                elif ae_vec_flipped[row, channel] == 0:
                    solver.add(local_entities[row * width * channels + 0 * channels + channel] == False)
                else:
                    raise ValueError("Should be 0 or 1, got {}".format(ae_vec_flipped[row, channel]))
        
        # sat?
        return solver.check() == sat
    
    def check_learned_ap_vector(self, data: List[Tuple[State, Set[GroundAtom], State, Set[GroundAtom], _Option, str]], \
                                curr_candidate: Predicate,
                                final_ae_vec: torch.Tensor) -> torch.Tensor:
        logging.info("Checking the learned AP vector...")
        logging.info("Using the provided AE vector (Add): {}".format(final_ae_vec[:, 0]))
        if final_ae_vec.shape[1] == 2:
            logging.info("Using the provided AE vector (Del): {}".format(final_ae_vec[:, 1]))
        ap_vec = torch.zeros([len(self.ae_row_names), 2]) # 0: exists, 1: not exists

        for s, _, s_, _, opt, opt_info in data:
            action_id = opt_info.split('_')[0]
            action_id = int(action_id)
            act_objs = opt.objects
            pred_types = curr_candidate.types
            ground_objs = [None] * len(pred_types)
            for obj in act_objs:
                if obj.type in pred_types:
                    pred_param_idx = pred_types.index(obj.type)
                    ground_objs[pred_param_idx] = obj
            t_value = int(curr_candidate.holds(s, ground_objs))
            t_value_ = int(curr_candidate.holds(s_, ground_objs))
            if self.equiv_effect(t_value, t_value_, final_ae_vec[action_id]):
                # precondition
                if t_value:
                    ap_vec[action_id, 0] += 1
                else:
                    ap_vec[action_id, 1] += 1
            else:
                logging.info("WARNING: Found sample that is not equivalent to AE Vector")
    
        return self.precondition_checker(ap_vec, final_ae_vec)
    
    def equiv_effect(self, t_value, t_value_, ae_vec) -> bool:
        if t_value == t_value_:
            # no change
            return ae_vec.all() == 0
        elif t_value == t_value_-1:
            # add effect
            return ae_vec[0] > 0.9
        elif t_value == t_value_+1:
            # delete effect
            return ae_vec[1] > 0.9
        else:
            raise ValueError("Should be one of the three cases")
        
    def _select_predicates_by_condition(self, candidates: Dict[Predicate, float], \
                                        dataset: Dataset) -> None:
        """Select a subset of the predicates to keep, using the consistency
        condition. The selected predicates are stored in self.learned_predicates.
        """
        data, trajectories = self._generate_data_from_dataset(dataset)
        self.initialize_ae_constraints(data, CFG.neupi_ae_matrix_channel)

        # Split the candidate by their types
        logging.info("*********************Splitting the candidates by their types*********************")
        for candidate_pred in candidates.keys():
            types = candidate_pred.types
            abandoned = True
            for typed_pred in self.learned_ae_pred_info.keys():
                if self.learned_ae_pred_info[typed_pred]['provided']:
                    continue
                belong_to_same_type = True
                if len(typed_pred.types) != len(types):
                    belong_to_same_type = False
                else:
                    for i in range(len(types)):
                        if typed_pred.types[i] != types[i]:
                            belong_to_same_type = False
                            break
                if belong_to_same_type:
                    self.learned_ae_pred_info[typed_pred]['all_candidates'].append(candidate_pred)
                    self.learned_ae_pred_info[typed_pred]['all_cost'].append(candidates[candidate_pred])
                    abandoned = False
                    break
            if abandoned:
                logging.info(f"Abandoning candidate {candidate_pred.name} since it does not belong to any type!")
            else:
                logging.info(f"Adding candidate {candidate_pred.name} to the corresponding type!")

        # Start Checking for the typed columns one by one
        for curr_pred in list(self.learned_ae_pred_info.keys()):
            logging.info(f"*********************Checking the Typed Predicate {curr_pred.name}*********************")
            if self.learned_ae_pred_info[curr_pred]['provided']:
                # this is learned/provided, skip
                logging.info(f"Skipping learning for {curr_pred.name} since it is already provided! Generate vectors for it!")
                ae_vectors = self.gen_sat_vec(curr_pred, 1, 100)
                assert len(ae_vectors) == 1, "Should find exactly one vector for provided predicate"
                ae_vector = ae_vectors[0]
                logging.info(f"AE Vector (Add): {ae_vector[:, 0]}")
                if ae_vector.shape[1] == 2:
                    logging.info(f"AE Vector (Del): {ae_vector[:, 1]}")
                self.learned_ae_pred_info[curr_pred]['ae_vecs'].append(ae_vector.clone())
                continue
            for config in self.pred_config:
                if config["name"] == curr_pred.name:
                    pred_config = config
            for curr_candidate in self.learned_ae_pred_info[curr_pred]['all_candidates']:
                logging.info(f"Checking consistent and informative condition of {curr_candidate.name}!")
                # Generate the vectors for the current candidate
                ae_vector = extract_ae_vector(data, curr_candidate, ae_row_name_dict=self.ae_row_names_dict)
                assert ae_vector.shape[-1] == CFG.neupi_ae_matrix_channel + 1, \
                    "The number of channels in AE matrix is not correct!"
                learned_guidance = compute_guidance_vector(
                                        ae_vector,
                                        min_prob=CFG.neupi_entropy_entry_min,
                                        max_prob=CFG.neupi_entropy_entry_max)
                if learned_guidance.sum() < pred_config["guidance_thresh"]:
                    logging.info(f"Consistency Checker 1 Passed for {curr_candidate.name}!")
                    # convert the ae_vector to the right shape
                    cate_vector_max_indices = torch.argmax(ae_vector, dim=1)
                    # from cate to n_row x n_channel
                    ae_vector = torch.zeros((len(self.ae_row_names), \
                                                  CFG.neupi_ae_matrix_channel), dtype=int)
                    if CFG.neupi_ae_matrix_channel == 1:
                        ae_vector = cate_vector_max_indices.clone()
                    else:
                        assert CFG.neupi_ae_matrix_channel == 2, "Only support 1 or 2 channels"
                        possible_values = torch.tensor([[0, 0], [1, 0], [0, 1]])
                        ae_vector = possible_values[cate_vector_max_indices].clone()
                    consistency_checker_2 = self.check_sat_vec(curr_pred, ae_vector.clone())
                    if consistency_checker_2:
                        logging.info(f"Consistency Checker 2 Also Passed for {curr_candidate.name}!")
                        precond_pass = self.check_learned_ap_vector(data, curr_candidate, \
                            ae_vector.clone())
                        if precond_pass:
                            logging.info(f"Precondition Checker Passed for {curr_candidate.name}!")
                            self.learned_ae_pred_info[curr_pred]['ae_vecs'].append(ae_vector.clone())
                            self.learned_ae_pred_info[curr_pred]['scores'].append(learned_guidance.clone())
                            self.learned_ae_pred_info[curr_pred]['model_weights'].append(curr_candidate)
                            logging.info(f"Adding {curr_candidate.name} to the final set of predicates!")
                        else:
                            logging.info(f"Precondition Checker Failed for {curr_candidate.name}!")
                    else:
                        logging.info(f"Consistency Checker 2 Failed for {curr_candidate.name}!")
                else:
                    logging.info(f"Consistency Checker 1 Failed for {curr_candidate.name}!")
            self.learned_ae_pred_info[curr_pred]['learned'] = True
            logging.info(f"Finished learning for {curr_pred.name}! Adding {len(self.learned_ae_pred_info[curr_pred]['ae_vecs'])} candidates to the final set of predicates!")
            
    def search_ae_matrix(self) -> Tuple[List[torch.Tensor], List[Set[Predicate]]]:
        # iterate all the possible combinate of the learned ae vector that satisfy the row constraints
        # order them in number of columns
        logging.info("Searching for AE Matrix...")
        colind2pred = {}
        ae_matrix_provided = []
        huge_ae_matrix_invented = []
        ae_col_names = []
        for pred in self.learned_ae_pred_info.keys():
            assert self.learned_ae_pred_info[pred]['learned']
            ae_vecs = self.learned_ae_pred_info[pred]['ae_vecs']
            if self.learned_ae_pred_info[pred]['provided']:
                ae_vec = ae_vecs[0]
                ae_matrix_provided.append(ae_vec.clone())
                ae_col_names.append(pred)
                continue
            for i, ae_vec in enumerate(ae_vecs):
                huge_ae_matrix_invented.append(ae_vec.clone())
                colind2pred[len(huge_ae_matrix_invented)-1] = \
                    self.learned_ae_pred_info[pred]['model_weights'][i]

        huge_ae_matrix_invented = torch.stack(huge_ae_matrix_invented, dim=1)
        ae_matrix_provided = torch.stack(ae_matrix_provided, dim=1)
        total_cols = huge_ae_matrix_invented.shape[0]
        possible_ae_matrix = []
        selected_invent_columns = []
        for k in range(1, total_cols + 1):
            logging.info(f"***********************Processing k={k}***********************")
            for selected_matrix, selected_indices in \
                tqdm(select_columns(huge_ae_matrix_invented, ae_matrix_provided, k)):
                # check if the selected matrix satisfy the row constraints
                if self.check_matrix_consistency(selected_matrix):
                    # add the selected matrix to the possible ae matrix
                    logging.info(f"Found a possible AE Matrix with {k} columns!")
                    possible_ae_matrix.append(selected_matrix)
                    selected_invent_columns.append(selected_indices)
                    logging.info(f"Current num of matrix: {len(possible_ae_matrix)}")

        predicate_sets = []
        used_predicates = []
        for selected_indices in selected_invent_columns:
            ae_col_names_curr = copy.deepcopy(ae_col_names)
            for idx in selected_indices:
                assert idx in colind2pred, "Should be in the dict"
                ae_col_names_curr.append(colind2pred[idx])
                if colind2pred[idx] not in used_predicates:
                    used_predicates.append(colind2pred[idx])
            predicate_sets.append(ae_col_names_curr)
        
        return possible_ae_matrix, predicate_sets, used_predicates
    
    def learn_from_offline_dataset(self, dataset: MDPDataset) -> None:
        original_dataset = mdp2positive(dataset)
        # Stpe 1: Generate a dataset of atom trajectories with all the candidate grammar
        atom_dataset, candidates = self._generate_atom_dataset_via_grammar(
            original_dataset)
        # Step 2: Select a subset of the candidates to keep, using consistency condition. Stored in
        # self.learned_ae_pred_info
        start = t.time()
        if CFG.grammar_search_use_condition:
            logging.info("Selecting predicates by condition...")
            self._select_predicates_by_condition(candidates, dataset)
            # Step 3: Combine the learned AE vectors to form the AE matrixes
            possible_ae_matrix, matrix_predicates, invented_predicates \
                    = self.search_ae_matrix()
            # Select a subset of the candidates to keep.
            # 2. Score Function Def
            score_function = _OperatorBeliefScoreFunction(
                    self._initial_predicates,
                    atom_dataset, self._train_tasks, self.ae_row_names,
                    CFG.neupi_aaai_metric)
            self._learned_predicates, self.ae_matrix_tgt, self.ae_col_names_all = \
                    self._select_predicates_by_greedy_score_search(
                    possible_ae_matrix, matrix_predicates, score_function)
            logging.info("Done.")
            logging.info("Final AE matrix (Add): {}".format(self.ae_matrix_tgt[:, :, 0]))
            if self.ae_matrix_tgt.shape[-1] == 2:
                logging.info("Final AE matrix (Del): {}".format(self.ae_matrix_tgt[:, :, 1]))
            logging.info("Final (Ordered) Predicates: {}".format(self.ae_col_names_all))
            logging.info("Time taken for the whole process: {}".format(t.time() - start))
            # Step 3: Compute the AAAI Objective for each matrix/predicate set
            logging.info("Learning NSRT from the learned predicates.")
            purned_atom_dataset = utils.prune_ground_atom_dataset(
                atom_dataset,
                self._learned_predicates | self._initial_predicates)
            # Finally, learn NSRTs via superclass, using all the kept predicates.
            annotations = None
            if dataset.has_annotations:
                annotations = dataset.annotations
            self._learn_nsrts(original_dataset.trajectories,
                            purned_atom_dataset,
                            annotations)
                            
        else:
            logging.info("Selecting a subset using search...")
            assert CFG.strips_learner == "cluster_and_intersect"
            if CFG.grammar_search_pred_selection_approach == "score_optimization":
                # Create the score function that will be used to guide search.
                score_function = create_score_function(
                    CFG.grammar_search_score_function, self._initial_predicates,
                    atom_dataset, candidates, self._train_tasks)
                self._learned_predicates = \
                    self._select_predicates_by_score_hillclimbing(
                    candidates, score_function, self._initial_predicates,
                    atom_dataset, self._train_tasks)
            elif CFG.grammar_search_pred_selection_approach == "clustering":
                self._learned_predicates = self._select_predicates_by_clustering(
                    candidates, self._initial_predicates, dataset, atom_dataset)
            logging.info("Time taken for the whole process: {}".format(t.time() - start))
            logging.info("Done.")
            # Finally, learn NSRTs via superclass, using all the kept predicates.
            annotations = None
            if dataset.has_annotations:
                annotations = dataset.annotations
            self._learn_nsrts(original_dataset.trajectories,
                            online_learning_cycle=None,
                            annotations=annotations)

    def _select_predicates_by_score_hillclimbing(
            self, candidates: Dict[Predicate, float],
            score_function: _PredicateSearchScoreFunction,
            initial_predicates: Set[Predicate],
            atom_dataset: List[GroundAtomTrajectory],
            train_tasks: List[Task]) -> Set[Predicate]:
        """Perform a greedy search over predicate sets."""

        # There are no goal states for this search; run until exhausted.
        def _check_goal(s: FrozenSet[Predicate]) -> bool:
            del s  # unused
            return False

        # Successively consider larger predicate sets.
        def _get_successors(
            s: FrozenSet[Predicate]
        ) -> Iterator[Tuple[None, FrozenSet[Predicate], float]]:
            for predicate in sorted(set(candidates) - s):  # determinism
                # Actions not needed. Frozensets for hashing. The cost of
                # 1.0 is irrelevant because we're doing GBFS / hill
                # climbing and not A* (because we don't care about the
                # path).
                yield (None, frozenset(s | {predicate}), 1.0)

        # Start the search with no candidates.
        init: FrozenSet[Predicate] = frozenset()

        # Greedy local hill climbing search.
        if CFG.grammar_search_search_algorithm == "hill_climbing":
            path, _, heuristics = utils.run_hill_climbing(
                init,
                _check_goal,
                _get_successors,
                score_function.evaluate,
                enforced_depth=CFG.grammar_search_hill_climbing_depth,
                parallelize=CFG.grammar_search_parallelize_hill_climbing)
            logging.info("\nHill climbing summary:")
            for i in range(1, len(path)):
                new_additions = path[i] - path[i - 1]
                assert len(new_additions) == 1
                new_addition = next(iter(new_additions))
                h = heuristics[i]
                prev_h = heuristics[i - 1]
                logging.info(f"\tOn step {i}, added {new_addition}, with "
                             f"heuristic {h:.3f} (an improvement of "
                             f"{prev_h - h:.3f} over the previous step)")
        elif CFG.grammar_search_search_algorithm == "gbfs":
            path, _ = utils.run_gbfs(
                init,
                _check_goal,
                _get_successors,
                score_function.evaluate,
                max_evals=CFG.grammar_search_gbfs_num_evals)
        else:
            raise NotImplementedError(
                "Unrecognized grammar_search_search_algorithm: "
                f"{CFG.grammar_search_search_algorithm}.")
        kept_predicates = path[-1]
        # The total number of predicate sets evaluated is just the
        # ((number of candidates selected) + 1) * total number of candidates.
        # However, since 'path' always has length one more than the
        # number of selected candidates (since it evaluates the empty
        # predicate set first), we can just compute it as below.
        assert self._metrics.get("total_num_predicate_evaluations") is None
        self._metrics["total_num_predicate_evaluations"] = len(path) * len(
            candidates)

        # Filter out predicates that don't appear in some operator
        # preconditions.
        logging.info("\nFiltering out predicates that don't appear in "
                     "preconditions...")
        preds = kept_predicates | initial_predicates
        pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, preds)
        segmented_trajs = [
            segment_trajectory(ll_traj, set(preds), atom_seq=atom_seq)
            for (ll_traj, atom_seq) in pruned_atom_data
        ]
        low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
        preds_in_preconds = set()
        for pnad in learn_strips_operators(low_level_trajs,
                                           train_tasks,
                                           set(kept_predicates
                                               | initial_predicates),
                                           segmented_trajs,
                                           verify_harmlessness=False,
                                           annotations=None,
                                           verbose=False):
            for atom in pnad.op.preconditions:
                preds_in_preconds.add(atom.predicate)
        kept_predicates &= preds_in_preconds

        logging.info(f"\nSelected {len(kept_predicates)} predicates out of "
                     f"{len(candidates)} candidates:")
        for pred in kept_predicates:
            logging.info(f"\t{pred}")
        score_function.evaluate(kept_predicates)  # log useful numbers

        return set(kept_predicates)

    @staticmethod
    def _get_consistent_predicates(
        predicates: Set[Predicate], clusters: List[List[Segment]]
    ) -> Tuple[Set[Predicate], Set[Predicate]]:
        """Returns all predicates that are consistent with respect to a set of
        segment clusters.

        A consistent predicate is is either an add effect, a delete
        effect, or doesn't change, within each cluster, for all
        clusters.
        """

        consistent: Set[Predicate] = set()
        inconsistent: Set[Predicate] = set()
        for pred in predicates:
            keep_pred = True
            for seg_list in clusters:
                segment_0 = seg_list[0]
                pred_in_add_effs_0 = pred in [
                    atom.predicate for atom in segment_0.add_effects
                ]
                pred_in_del_effs_0 = pred in [
                    atom.predicate for atom in segment_0.delete_effects
                ]
                for seg in seg_list[1:]:
                    pred_in_curr_add_effs = pred in [
                        atom.predicate for atom in seg.add_effects
                    ]
                    pred_in_curr_del_effs = pred in [
                        atom.predicate for atom in seg.delete_effects
                    ]
                    not_consis_add = pred_in_add_effs_0 != pred_in_curr_add_effs
                    not_consis_del = pred_in_del_effs_0 != pred_in_curr_del_effs
                    if not_consis_add or not_consis_del:
                        keep_pred = False
                        inconsistent.add(pred)
                        logging.info(f"Inconsistent predicate: {pred.name}")
                        break
                if not keep_pred:
                    break
            else:
                consistent.add(pred)
        return consistent, inconsistent

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int],
                     annotations: Optional[List[Any]]) -> None:
        dataset_fname, _ = utils.create_dataset_filename_str(
            saving_ground_atoms=True,
            online_learning_cycle=online_learning_cycle)
        # If CFG.load_atoms is set, then try to create a GroundAtomTrajectory
        # by loading sets of GroundAtoms directly from a saved file.
        # By default, we don't create a full ground atom dataset, since
        # doing so requires called abstract on all states, including states
        # that might ultimately just be in the middle of segments. When
        # options take many steps, this makes a big time/space difference.
        ground_atom_dataset: Optional[List[GroundAtomTrajectory]] = None
        if CFG.load_atoms:
            ground_atom_dataset = utils.load_ground_atom_dataset(
                dataset_fname, trajectories)
        elif CFG.save_atoms:
            # Apply predicates to data, producing a dataset of abstract states.
            ground_atom_dataset = utils.create_ground_atom_dataset(
                trajectories, self._get_current_predicates())
            utils.save_ground_atom_dataset(ground_atom_dataset, dataset_fname)
        elif CFG.offline_data_method in [
                "demo+labelled_atoms", "saved_vlm_img_demos_folder",
                "demo_with_vlm_imgs"
        ]:
            # In this case, the annotations are basically ground atoms!
            # We can use these to make GroundAtomTrajectories.
            assert annotations is not None
            assert len(annotations) == len(trajectories)
            ground_atom_dataset = []
            annotations_with_only_selected_preds = []
            selected_preds = self._get_current_predicates()
            for atoms_traj in annotations:
                curr_selected_preds_atoms_traj = []
                for atoms_set in atoms_traj:
                    curr_selected_preds_atoms_set = set(
                        atom for atom in atoms_set
                        if atom.predicate in selected_preds)
                    curr_selected_preds_atoms_traj.append(
                        curr_selected_preds_atoms_set)
                annotations_with_only_selected_preds.append(
                    curr_selected_preds_atoms_traj)
            for ll_traj, atoms in zip(trajectories,
                                      annotations_with_only_selected_preds):
                ground_atom_dataset.append((ll_traj, atoms))
        self._nsrts, self._segmented_trajs, self._seg_to_nsrt = \
            learn_nsrts_from_data(trajectories,
                                  self._train_tasks,
                                  self._get_current_predicates(),
                                  self._initial_options,
                                  self._action_space,
                                  ground_atom_dataset,
                                  sampler_learner=CFG.sampler_learner,
                                  annotations=annotations)
        # save_path = utils.get_approach_save_path_str()
        # with open(f"{save_path}_{online_learning_cycle}.NSRTs", "wb") as f:
        #     pkl.dump(self._nsrts, f)
        if CFG.compute_sidelining_objective_value:
            self._compute_sidelining_objective_value(trajectories)