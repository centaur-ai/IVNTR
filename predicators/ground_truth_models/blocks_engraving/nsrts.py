"""Ground-truth NSRTs for the blocks environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class BlocksEngraveGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the blocks environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"blocks_engrave_vec", "blocks_engrave_pcd", "blocks_engrave_pcdnorm"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["block"]
        robot_type = types["robot"]

        # Predicates
        On = predicates["On"]
        FaceUp = predicates["FaceUp"]
        FaceDown = predicates["FaceDown"]
        Matched = predicates["Matched"]
        Single = predicates["Single"]
        OnTable = predicates["OnTable"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        Clear = predicates["Clear"]
        NotEq = predicates["NotEq"]
        GoalAchieved = predicates["GoalAchieved"]

        # Options
        PickFromTable = options["PickFromTable"]
        Unstack = options["Unstack"]
        Stack = options["Stack"]
        PutOnTable = options["PutOnTable"]
        RotateDown = options["RotateDown"]
        # RotateUp = options["RotateUp"]
        Engrave = options["Engrave"]
        AchieveGoal = options["AchieveGoal"]

        nsrts = set()

        # PickFromTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [robot, block]
        option_vars = [robot, block]
        option = PickFromTable
        preconditions = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(Holding, [robot, block])}
        delete_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(GripperOpen, [robot])
        }

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
        preconditions = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(NotEq, [block, otherblock]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(Clear, [otherblock])
        }
        delete_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(GripperOpen, [robot])
        }
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
        preconditions = {
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(Clear, [otherblock]),
            LiftedAtom(NotEq, [block, otherblock]),
        }
        add_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(Clear, [otherblock])
        }

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
        preconditions = {
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(FaceUp, [block])
        }
        add_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {LiftedAtom(Holding, [robot, block])}

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

        # RotateDown
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [robot, block]
        option_vars = [robot, block]
        option = RotateDown
        preconditions = {
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(FaceUp, [block])
        }
        add_effects = {LiftedAtom(FaceDown, [block])}
        delete_effects = {LiftedAtom(FaceUp, [block])}

        rotatedown_nsrt = NSRT("RotateDown", parameters, preconditions,
                                 add_effects, delete_effects, set(), option,
                                 option_vars, null_sampler)
        nsrts.add(rotatedown_nsrt)

        # RotateUp, not used for now
        # block = Variable("?block", block_type)
        # robot = Variable("?robot", robot_type)
        # parameters = [robot, block]
        # option_vars = [robot, block]
        # option = RotateUp
        # preconditions = {
        #     LiftedAtom(Holding, [robot, block]),
        #     LiftedAtom(FaceDown, [block])
        # }
        # add_effects = {LiftedAtom(FaceUp, [block])}
        # delete_effects = {LiftedAtom(FaceDown, [block])}

        # rotateup_nsrt = NSRT("RotateUp", parameters, preconditions,
        #                          add_effects, delete_effects, set(), option,
        #                          option_vars, null_sampler)
        # nsrts.add(rotateup_nsrt)

        # Engrave
        top_block = Variable("?block", block_type)
        bottom_block = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [robot, top_block, bottom_block]
        option_vars = [robot, top_block, bottom_block]
        option = Engrave
        preconditions = {
            LiftedAtom(OnTable, [bottom_block]),
            LiftedAtom(On, [top_block, bottom_block]),
            LiftedAtom(Clear, [top_block]),
            LiftedAtom(Single, [top_block]),
            LiftedAtom(Single, [bottom_block]),
            LiftedAtom(FaceDown, [top_block]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(Matched, [top_block, bottom_block])}
        delete_effects = {LiftedAtom(Single, [block]), 
                          LiftedAtom(Single, [otherblock])}

        def engrave_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: normalized coordinates w.r.t. block size.
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        engrave_nsrt = NSRT("Engrave", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            engrave_sampler)
        nsrts.add(engrave_nsrt)

        # AchieveGoal
        robot = Variable("?robot", robot_type)
        top_block = Variable("?block", block_type)
        bottom_block = Variable("?otherblock", block_type)
        parameters = [robot, top_block, bottom_block]
        option_vars = [robot, top_block, bottom_block]
        option = AchieveGoal
        preconditions = {
            LiftedAtom(On, [top_block, bottom_block]),
            LiftedAtom(Matched, [top_block, bottom_block]),
            LiftedAtom(FaceDown, [top_block]),
            LiftedAtom(FaceUp, [bottom_block]),
            LiftedAtom(OnTable, [bottom_block]),
            LiftedAtom(Clear, [top_block]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {
            LiftedAtom(GoalAchieved, [top_block, bottom_block])
        }
        delete_effects = {}

        stack_nsrt = NSRT("AchieveGoal", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          null_sampler)
        nsrts.add(stack_nsrt)

        return nsrts
