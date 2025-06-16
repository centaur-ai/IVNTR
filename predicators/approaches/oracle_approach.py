"""A bilevel planning approach that uses hand-specified NSRTs.

The approach is aware of the initial predicates and options. Predicates
that are not in the initial predicates are excluded from the ground
truth NSRTs. If an NSRT's option is not included, that NSRT will not be
generated at all.
"""

from typing import List, Optional, Set, Callable
import logging
from gym.spaces import Box

from predicators import utils
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.approaches import ApproachFailure
from predicators.planning import _sesame_plan_for_mdp
from predicators.ground_truth_models import get_gt_nsrts
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, ParameterizedOption, Predicate, Task, \
    Type, State, Action


class OracleApproach(BilevelPlanningApproach):
    """A bilevel planning approach that uses hand-specified NSRTs."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 nsrts: Optional[Set[NSRT]] = None,
                 option_model: Optional[_OptionModelBase] = None) -> None:
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)
        if nsrts is None:
            nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
                                 self._initial_options)
        logging.info(f"Using {len(nsrts)} NSRTs: {nsrts}")
        self._sorted_options = sorted(self._initial_options,
                                      key=lambda o: o.name)
        if CFG.verbose_pddl_matrix:
            # Step 1: Understand Operator Types, aka, Rows
            self._obj_types = {}
            self._opt_type_id = {}
            for option in self._sorted_options:
                self._opt_type_id[option.name] = {}
                types = option.types
                for t in types:
                    if t.name not in self._obj_types:
                        self._obj_types[t.name] = t
                    if t.name not in self._opt_type_id[option.name]:
                        self._opt_type_id[option.name][t.name] = [0]
                    else:
                        curr_id = len(self._opt_type_id[option.name][t.name])
                        self._opt_type_id[option.name][t.name].append(curr_id)
            logging.info(f"Rows of the AE Matrix")
            for o in self._opt_type_id.keys():
                arguments = []
                for t in self._opt_type_id[o].keys():
                    for ent_id in self._opt_type_id[o][t]:
                        arguments.append(f"{t}{ent_id}")
                complete_name = f"{o}({', '.join(arguments)})"
                logging.info(f"{complete_name}")
            # Step 2: Understand Predicate Types, aka, Columns
            effect_predicates = {}
            for nsrt in nsrts:
                # note that we are using options as rows, not nsrts, they have different idx mapping
                option_index = self._sorted_options.index(nsrt.option)
                nsrt_params = nsrt.parameters
                all_type_count = {}
                nsrt_ent_ids = {}
                for param in nsrt_params:
                    if param.type.name not in all_type_count:
                        all_type_count[param.type.name] = 0
                    else:
                        all_type_count[param.type.name] += 1
                    nsrt_ent_ids[param] = all_type_count[param.type.name]
                add_effects = list(nsrt.add_effects)
                del_effects = list(nsrt.delete_effects)
                for lifted_pred in add_effects:
                    pred = lifted_pred.predicate
                    # collect the info for this lifted predicate
                    pred_dict = {
                            "effect_vec": [0] * len(nsrts),
                            "type": pred.types,
                            "ent_id": ['0'] * len(pred.types)
                        }
                    for param_id, param in enumerate(lifted_pred.variables):
                        pred_dict["ent_id"][param_id] \
                            = str(nsrt_ent_ids[param])
                    pred_dict["effect_vec"][option_index] = 1
                    # use pred name and ent idx to distinguish effect predicates
                    pred_vec_name = f"{pred.name}({', '.join(pred_dict['ent_id'])})"
                    # decide if we need to create a new column for it
                    if pred_vec_name not in effect_predicates:
                        # logging.info(f"Added new column for {pred_vec_name}")
                        effect_predicates[pred_vec_name] = pred_dict
                    else:
                        # logging.info(f"{pred_vec_name} already exists, its current")
                        # logging.info(f"vector is {effect_predicates[pred_vec_name]['effect_vec']}")
                        effect_predicates[pred_vec_name]['effect_vec'][option_index] = 1
                for lifted_pred in del_effects:
                    pred = lifted_pred.predicate
                    # collect the info for this lifted predicate
                    pred_dict = {
                            "effect_vec": [0] * len(nsrts),
                            "type": pred.types,
                            "ent_id": ['0'] * len(pred.types)
                        }
                    for param_id, param in enumerate(lifted_pred.variables):
                        pred_dict["ent_id"][param_id] \
                            = str(nsrt_ent_ids[param])
                    pred_dict["effect_vec"][option_index] = 2
                    # use pred name and ent idx to distinguish effect predicates
                    pred_vec_name = f"{pred.name}({', '.join(pred_dict['ent_id'])})"
                    # decide if we need to create a new column for it
                    if pred_vec_name not in effect_predicates:
                        # logging.info(f"Added new column for {pred_vec_name}")
                        effect_predicates[pred_vec_name] = pred_dict
                    else:
                        # logging.info(f"{pred_vec_name} already exists, its current")
                        # logging.info(f"vector is {effect_predicates[pred_vec_name]['effect_vec']}")
                        effect_predicates[pred_vec_name]["effect_vec"][option_index] = 2
            # Step 3: Print out the effect predicates
            logging.info(f"Name and Columns of the AE Matrix (1: Add, 2: Del):")
            for pred_name in effect_predicates.keys():
                logging.info(f"{pred_name} ({effect_predicates[pred_name]['type']}): {effect_predicates[pred_name]['effect_vec']}")
                
        self._nsrts = nsrts
        self._pos_end_nodes_mdp = {}
        self._neg_end_nodes_mdp = {}

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts
    
    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        if 'reward' not in CFG.offline_data_method:
            return super()._solve(task, timeout)
        else:
            # collect demonstration data with its original graph and reward
            logging.info("Collecting demonstration data with original graph and reward")
            logging.info("In this mode, we set hl search timeout to {} seconds".format(CFG.mdp_hl_search_timeout))
            timeout = CFG.mdp_hl_search_timeout
            logging.info(f"Besides, we will (try to) search for {CFG.all_skeleton_found} (optimal) positive skeletons in A*.")
            self._num_calls += 1
            self._max_skeletons_optimized = CFG.all_skeleton_found
            # ensure random over successive calls
            seed = self._seed + self._num_calls
            nsrts = self._get_current_nsrts()
            preds = self._get_current_predicates()

            # Run full bilevel planning.
            # pos_end_nodes_mdp, neg_end_nodes_mdp
            # randomly select a plan from pos_mdp
            pos_end_nodes_mdp, neg_end_nodes_mdp = _sesame_plan_for_mdp(
                            task,
                            self._option_model,
                            nsrts,
                            preds,
                            self._types,
                            timeout,
                            seed,
                            self._task_planning_heuristic,
                            self._max_skeletons_optimized,
                            max_horizon=CFG.horizon,
                            allow_noops=CFG.sesame_allow_noops,
                            use_visited_state_set=CFG.sesame_use_visited_state_set)
            
            pos_mdp_id = self._rng.choice(list(pos_end_nodes_mdp.keys()))
            option_plan, nsrt_plan, rw, metrics = pos_end_nodes_mdp[pos_mdp_id]
            self._last_plan = option_plan
            self._last_nsrt_plan = nsrt_plan
            policy = utils.option_plan_to_policy(option_plan)

            self._save_metrics(metrics, nsrts, preds)
            self._save_mdp_data(pos_end_nodes_mdp, neg_end_nodes_mdp)

            def _policy(s: State) -> Action:
                try:
                    return policy(s)
                except utils.OptionExecutionFailure as e:
                    raise ApproachFailure(e.args[0], e.info)

            return _policy
        
    def _save_mdp_data(self, pos_end_nodes_mdp, neg_end_nodes_mdp):
        self._pos_end_nodes_mdp = pos_end_nodes_mdp
        self._neg_end_nodes_mdp = neg_end_nodes_mdp

    def get_all_mdps(self):
        return self._pos_end_nodes_mdp, self._neg_end_nodes_mdp
