"""An approach that just executes random options."""

from typing import Callable, List, Set
import os
import time
import dill as pkl
import logging
from predicators import utils
from predicators.ground_truth_models import get_dummy_nsrts
from predicators.approaches import ApproachFailure, BaseApproach, ApproachTimeout
from predicators.structs import Action, State, Task
from predicators.option_model import create_option_model
from gym.spaces import Box
from predicators.settings import CFG
from predicators.planning import task_plan_grounding
from predicators.structs import Action, ParameterizedOption, Predicate, State, Task, \
    Type, _Option, NSRT
from predicators.nsrt_learning.sampler_learning import _create_neural_sampler


class RandomNSRTApproach(BaseApproach):
    """Samples random dummy NSRTs (and so the params are obtained by GT sampler)."""
    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._sorted_options = sorted(self._initial_options,
                                        key=lambda o: o.name)
        self._option_model = create_option_model(CFG.option_model_name)
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

    @classmethod
    def get_name(cls) -> str:
        return "random_nsrt"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        if CFG.random_policy_solve_with_shooting:
            return self._solve_with_shooting(task, timeout)
        return self._solve_without_shooting(task)
    
    def _solve_without_shooting(self, task: Task, timeout: int) -> Callable[[State], Action]:

        def fallback_policy(state: State) -> Action:
            del state  # unused
            raise ApproachFailure("Random nsrt sampling failed!")
        init_atoms = utils.abstract(task.init, self._initial_predicates)
        objects = set(task.init)
        # all ground nsrts, even though their effects are empty
        ground_nsrts, _ = task_plan_grounding(
            init_atoms, objects, self._dummy_nsrts, allow_noops=True)

        return utils.create_random_nsrt_policy(ground_nsrts,
                                            self._initial_predicates,
                                            task.goal,
                                            self._rng, fallback_policy)

    def _solve_with_shooting(self, task: Task,
                             timeout: int) -> Callable[[State], Action]:
        # has simulator, shooting + checking until timeout
        start_time = time.perf_counter()
        goal = task.goal
        # Keep trying until the timeout.
        tries: int = 0
        all_num_act: int = 0
        total_num_act: int = 0
        init_atoms = utils.abstract(task.init, self._initial_predicates)
        objects = set(task.init)
        # all possible nsrts in this grounded task
        ground_nsrts, _ = task_plan_grounding(
            init_atoms, objects, self._dummy_nsrts, allow_noops=True)
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
                # Sample an initiable option.
                sample_nsrt = utils.sample_applicable_ground_nsrt(state, ground_nsrts, \
                                            self._initial_predicates, self._rng)
                if sample_nsrt is None:
                    break
                cur_option = None
                for i in range(CFG.random_policy_shooting_max_samples):
                    try:
                        cur_option = sample_nsrt.sample_option(state,
                                                    goal=goal,
                                                    rng=self._rng)
                    except AssertionError:
                        continue
                    if cur_option.initiable(state):
                        break
                if cur_option is None:
                    break
                plan.append(cur_option)
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
        raise ApproachTimeout("Shooting timed out!",
                              info={
                                  "partial_refinements": (None, plan),
                              })