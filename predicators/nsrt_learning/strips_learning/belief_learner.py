"""Algorithms for STRIPS learning that rely on clustering to obtain effects."""

import abc
import functools
import logging
from collections import defaultdict
from typing import Dict, FrozenSet, Iterator, List, Set, Tuple, cast

from predicators import utils
from predicators.gnn.neupi_utils import parse_basic_pred_name
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, Datastore, DummyOption, LiftedAtom, \
    ParameterizedOption, Predicate, STRIPSOperator, VarToObjSub, Variable


class BeliefSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a clustering-based STRIPS learner."""

    def _learn(self) -> List[PNAD]:
        num_sample_in = 0
        num_sample_out = 0
        segments = [seg for segs in self._segmented_trajs for seg in segs]
        # Cluster the segments according to common option and effects.
        pnads, added_segment_idxs = self._belief2pnads_init(segments)
        filtered_pnads = []
        for pnad in pnads:
            # Try to unify this transition with existing effects.
            # Note that both add and delete effects must unify,
            # and also the objects that are arguments to the options.
            if len(pnad.datastore) > 0:
                filtered_pnads.append(pnad)
                continue
            else:
                logging.info(f"PNAD {pnad.op.name} has no samples. Learning Op failed.")
                return []

        for ids, segment in enumerate(segments):
            if ids in added_segment_idxs:
                # this segment has been added to a PNAD
                continue
            if segment.has_option():
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
            else:
                segment_param_option = DummyOption.parent
                segment_option_objs = tuple()
            for pnad in filtered_pnads:
                # Try to unify this transition with existing effects.
                # Note that both add and delete effects must unify,
                # and also the objects that are arguments to the options.
                (pnad_param_option, pnad_option_vars) = pnad.option_spec
                suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                    frozenset(),  # no preconditions
                    frozenset(),  # no preconditions
                    frozenset(segment.add_effects),
                    frozenset(pnad.op.add_effects),
                    frozenset(segment.delete_effects),
                    frozenset(pnad.op.delete_effects),
                    segment_param_option,
                    pnad_param_option,
                    segment_option_objs,
                    tuple(pnad_option_vars))
                sub = cast(VarToObjSub,
                           {v: o
                            for o, v in ent_to_ent_sub.items()})
                if suc:
                    # Add to this PNAD.
                    assert set(sub.keys()) == set(pnad.op.parameters)
                    pnad.add_to_datastore((segment, sub))
                    num_sample_in += 1
                    break
            if not suc:
                # the sample does not fit any existing PNAD
                num_sample_out += 1
        logging.info(f"Number of samples in: {num_sample_in}, Specifically:")
        for pnad in filtered_pnads:
            logging.info(f"Number of samples in for {pnad.op.name}: {len(pnad.datastore)}")
        logging.info(f"Number of samples out: {num_sample_out}")

        # Learn the preconditions of the operators in the PNADs. This part
        # is flexible; subclasses choose how to implement it.
        filtered_pnads = self._learn_pnad_preconditions(filtered_pnads)

        # Handle optional postprocessing to learn ignore effects.
        filtered_pnads = self._postprocessing_learn_ignore_effects(filtered_pnads)

        # Log and return the PNADs.
        if self._verbose:
            logging.info("Learned operators (before option learning):")
            for pnad in filtered_pnads:
                logging.info(pnad)
        return filtered_pnads

    @abc.abstractmethod
    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Subclass-specific algorithm for learning PNAD preconditions.

        Returns a list of new PNADs. Should NOT modify the given PNADs.
        """
        raise NotImplementedError("Override me!")

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        """Optionally postprocess to learn ignore effects."""
        _ = self  # unused, but may be used in subclasses
        return pnads
    
    def _get_pred_input_vars(self, pred: Predicate, col_ent_idx: List[int], \
                             opt_param: List[Variable]) -> List:
        """Get the input variables of a predicate."""
        # Update this function for same type inputs.
        opt_param_dict = {}
        for var in opt_param:
            if var.type.name not in opt_param_dict:
                opt_param_dict[var.type.name] = [var]
            else:
                opt_param_dict[var.type.name].append(var)
        input_vars = []
        for i, pt in enumerate(pred.types):
            n = col_ent_idx[i]
            if n >= len(opt_param_dict[pt.name]):
                input_vars.append(opt_param_dict[pt.name][0])
            else:
                input_vars.append(opt_param_dict[pt.name][n])
        return input_vars

    def _match_objs2vars(self, objects: List, params: List[Variable],
                        add_effects: Set[LiftedAtom], delete_effects: Set[LiftedAtom],
                        segment: Datastore
                        ) -> Tuple[bool, Dict, Dict]:
        """Match objects to variables by checking equivalent effects."""
        # Important: we assume objects are ordered in the same way as params
        from itertools import permutations, product

        # Step 1: Group variables and objects by their types.
        type_to_vars = {}
        type_to_objs = {}

        for var in params:
            type_to_vars.setdefault(var.type, []).append(var)

        for obj in objects:
            type_to_objs.setdefault(obj.type, []).append(obj)

        # Step 2: Check for type mismatches or insufficient objects.
        for type_ in type_to_vars:
            if type_ not in type_to_objs or len(type_to_vars[type_]) \
                > len(type_to_objs[type_]):
                # Not enough objects to match variables of this type.
                return False, {}, {}
        
        # Step 2.5: Check if the objects are already matched.
        full_var_to_obj_mapping = {}
        for var, obj in zip(params, objects):
            full_var_to_obj_mapping[var] = obj

        obj_to_var = {obj: var for var, obj in full_var_to_obj_mapping.items()}
        add_effects_seg = {
            atom.lift(obj_to_var)
            for atom in segment.add_effects
        }
        delete_effects_seg = {
            atom.lift(obj_to_var)
            for atom in segment.delete_effects
        }

        if add_effects_seg == add_effects and delete_effects_seg == delete_effects:
            # Successful mapping found.
            succ = True
            var_to_obj = full_var_to_obj_mapping
            return succ, obj_to_var, var_to_obj

        logging.info(f"Objects are not already matched. Try to match them...")
        # Step 3: Generate all possible mappings for each type.
        mappings_per_type = []

        for type_ in type_to_vars:
            vars_list = type_to_vars[type_]
            objs_list = type_to_objs[type_]

            # Generate all permutations of objects assigned to variables.
            type_mappings = [
                dict(zip(vars_list, objs_perm))
                for objs_perm in permutations(objs_list, len(vars_list))
            ]
            mappings_per_type.append(type_mappings)

        # Step 4: Generate all combinations of type-specific mappings.
        all_mappings = product(*mappings_per_type)

        # Step 5: Test each combined mapping.
        for mapping_combination in all_mappings:
            full_var_to_obj_mapping = {}
            used_objs = set()
            valid_mapping = True

            # Combine mappings from all types.
            for mapping in mapping_combination:
                for var, obj in mapping.items():
                    if obj in used_objs:
                        # Object already assigned; skip this mapping.
                        valid_mapping = False
                        break
                    full_var_to_obj_mapping[var] = obj
                    used_objs.add(obj)
                if not valid_mapping:
                    break

            if not valid_mapping:
                continue  # Try the next mapping combination.

            # Step 6: Create object-to-variable mapping.
            obj_to_var = {obj: var for var, obj in full_var_to_obj_mapping.items()}

            # Step 7: Lift the segment effects using the current mapping.
            add_effects_seg = {
                atom.lift(obj_to_var)
                for atom in segment.add_effects
            }
            delete_effects_seg = {
                atom.lift(obj_to_var)
                for atom in segment.delete_effects
            }

            # Step 8: Check if the lifted effects match the target effects.
            if add_effects_seg == add_effects and delete_effects_seg == delete_effects:
                # Successful mapping found.
                succ = True
                var_to_obj = full_var_to_obj_mapping
                return succ, obj_to_var, var_to_obj

        # No valid mapping found after exhausting all possibilities.
        return False, {}, {}

    def _belief2pnads_init(self, segments: List) -> List[PNAD]:
        """Initialize the PNADs with the belief that each segment is a new
        operator."""
        pnads: List[PNAD] = []
        assert self._belief, "Belief must be set before calling this method."
        row_names = self._belief["row_names"]
        col_names = self._belief["col_names"]
        col_ent_idx = self._belief["col_ent_idx"]
        ae_matrix = self._belief["ae_matrix"]
        logging.info(f"Constructing PNADs from belief AE matrix Tgt.")
        added_segment_idxs = []
        for i, option in enumerate(row_names):
            logging.info(f"Constructing PNAD for Row {i}, option {option}.")
            params = utils.create_new_variables(
                    [t for t in option.types])
            preconds: Set[LiftedAtom] = set()  # will be learned later
            add_effects: Set[LiftedAtom] = set()
            delete_effects: Set[LiftedAtom] = set()
            for j, effect_p in enumerate(col_names):
                if ae_matrix[i, j].sum() == 0:
                    continue
                else:
                    assert ae_matrix[i, j].sum() == 1
                    input_vars = self._get_pred_input_vars(effect_p, col_ent_idx[j], params)
                    if effect_p.arity == 0:
                        # no arguments
                        lifted_atom = LiftedAtom(effect_p, [])
                    else:
                        lifted_atom = effect_p(input_vars)
                    if ae_matrix[i, j, 0] == 1:
                        add_effects.add(lifted_atom)
                    else:
                        delete_effects.add(lifted_atom)
            # every operator has non-empty effects
            # assert add_effects or delete_effects
            ignore_effects: Set[Predicate] = set()  # will be learned later
            op = STRIPSOperator(f"Op{i}", params, preconds,
                                    add_effects, delete_effects,
                                    ignore_effects)
            # Find a segment that has the same option and effect
            datastore: List = []
            option_spec: Tuple = ()
            for ids, segment in enumerate(segments):
                if segment.has_option():
                    segment_option = segment.get_option()
                    segment_param_option = segment_option.parent
                    segment_option_objs = tuple(segment_option.objects)
                else:
                    segment_param_option = DummyOption.parent
                    segment_option_objs = tuple()
                if segment_param_option == option:
                    # Try to see if the effects match
                    effect_objects = sorted({o for atom in segment.add_effects |
                               segment.delete_effects for o in atom.objects} | \
                              set(segment_option_objs))
                    objects_lst = segment._option.objects
                    if len(objects_lst) != len(params):
                        # this can't be the right segment
                        continue
                    if len(effect_objects) != len(params):
                        # this can't be the right segment
                        # as the effects involves more objects not operated
                        continue
                    succ, obj_to_var, var_to_obj = \
                            self._match_objs2vars(objects_lst, params, \
                                        add_effects, delete_effects, \
                                        segment)
                    if succ:
                        datastore.append((segment, var_to_obj))
                        option_vars = [obj_to_var[o] for o in segment_option_objs]
                        option_spec = (segment_param_option, option_vars)
                        # we got the right segment
                        added_segment_idxs.append(ids)
                        break
            pnads.append(PNAD(op, datastore, option_spec))
        return pnads, added_segment_idxs





class BeliefAndIntersectSTRIPSLearner(BeliefSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via
    intersection."""

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        new_pnads = []
        for pnad in pnads:
            preconditions = self._induce_preconditions_via_intersection(pnad)
            # Since we are taking an intersection, we're guaranteed that the
            # datastore can't change, so we can safely use pnad.datastore here.
            new_pnads.append(
                PNAD(pnad.op.copy_with(preconditions=preconditions),
                     pnad.datastore, pnad.option_spec))
        return new_pnads

    @classmethod
    def get_name(cls) -> str:
        return "belief_and_intersect"

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        """Prune PNADs whose datastores are too small.

        Specifically, keep PNADs that have at least
        CFG.cluster_and_intersect_min_datastore_fraction fraction of the
        segments produced by the option in their NSRT.
        """
        if not CFG.cluster_and_intersect_prune_low_data_pnads:
            return pnads
        option_to_dataset_size: Dict[ParameterizedOption,
                                     int] = defaultdict(int)
        for pnad in pnads:
            option = pnad.option_spec[0]
            option_to_dataset_size[option] += len(pnad.datastore)
        ret_pnads: List[PNAD] = []
        for pnad in pnads:
            option = pnad.option_spec[0]
            fraction = len(pnad.datastore) / option_to_dataset_size[option]
            if fraction >= CFG.cluster_and_intersect_min_datastore_fraction:
                ret_pnads.append(pnad)
        return ret_pnads