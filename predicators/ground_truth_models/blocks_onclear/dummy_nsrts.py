"""Ground-truth NSRTs for the blocks environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import DummyNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class BlocksOnClearDummyNSRTFactory(DummyNSRTFactory):
    """Ground-truth NSRTs for the blocks environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"blocks_onclear"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["block"]
        robot_type = types["robot"]

        # Predicates
        GoalAchieved = predicates["GoalAchieved"]

        # Options
        PickFromTable = options["PickFromTable"]
        Unstack = options["Unstack"]
        Stack = options["Stack"]
        PutOnTable = options["PutOnTable"]
        AchieveGoal = options["AchieveGoal"]

        nsrts = set()

        # PickFromTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [robot, block]
        option_vars = [robot, block]
        option = PickFromTable
        preconditions = set()
        add_effects = set()
        delete_effects = set()

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)
        nsrts.add(pickfromtable_nsrt)

        # Unstack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [robot, block, otherblock]
        option_vars = [robot, block, otherblock]
        option = Unstack
        preconditions = set()
        add_effects = set()
        delete_effects = set()
        unstack_nsrt = NSRT("Unstack", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            null_sampler)
        nsrts.add(unstack_nsrt)

        # Stack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [robot, block, otherblock]
        option_vars = [robot, block, otherblock]
        option = Stack
        preconditions = set()
        add_effects = set()
        delete_effects = set()

        stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          null_sampler)
        nsrts.add(stack_nsrt)

        # PutOnTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [robot, block]
        option_vars = [robot, block]
        option = PutOnTable
        preconditions = set()
        add_effects = set()
        delete_effects = set()

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: normalized coordinates w.r.t. workspace.
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putontable_sampler)
        nsrts.add(putontable_nsrt)

        # AchieveGoal
        top_block = Variable("?block", block_type)
        bottom_block = Variable("?otherblock", block_type)
        parameters = [top_block, bottom_block]
        option_vars = [top_block, bottom_block]
        option = AchieveGoal
        preconditions = set()
        add_effects = {
            LiftedAtom(GoalAchieved, [top_block, bottom_block])
        }
        delete_effects = {}

        stack_nsrt = NSRT("AchieveGoal", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          null_sampler)
        nsrts.add(stack_nsrt)

        return nsrts
