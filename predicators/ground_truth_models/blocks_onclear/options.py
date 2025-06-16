"""Ground-truth options for the (non-pybullet) blocks environment."""

from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.blocks import BlocksEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class BlocksOnClearGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the (non-pybullet) blocks environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"blocks_onclear"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        block_type = types["block"]
        block_size = CFG.blocks_block_size

        PickFromTable = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            # params: []
            "PickFromTable",
            cls._create_pickfromtabel_policy(action_space),
            types=[robot_type, block_type])
        
        Unstack = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick, object on which to stack currently-held-object]
            # params: []
            "Unstack",
            cls._create_unstack_policy(action_space),
            types=[robot_type, block_type, block_type])

        Stack = utils.SingletonParameterizedOption(
            # variables: [robot, currently-held-object, object on which to stack currently-held-object]
            # params: []
            "Stack",
            cls._create_stack_policy(action_space, block_size),
            types=[robot_type, block_type, block_type])

        PutOnTable = utils.SingletonParameterizedOption(
            # variables: [robot, currently-held-object]
            # params: [x, y] (normalized coordinates on the table surface)
            "PutOnTable",
            cls._create_putontable_policy(action_space, block_size),
            types=[robot_type, block_type],
            params_space=Box(0, 1, (2, )))
        
        AchieveGoal = utils.SingletonParameterizedOption(
            # variables: [top block, bottom block]
            # params: []
            "AchieveGoal",
            cls._create_achievegoal_policy(action_space),
            types=[block_type, block_type])

        return {PickFromTable, Unstack, Stack, PutOnTable, AchieveGoal}

    @classmethod
    def _create_pickfromtabel_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, block = objects
            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            arr = np.r_[block_pose, 0.0, 0.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy
    
    @classmethod
    def _create_unstack_policy(cls, action_space: Box) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, block, otherblock = objects
            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            # assert state.get(otherblock, "pose_z") < state.get(block, "pose_z"), \
            # "Block to unstack must be below the other block."
            arr = np.r_[block_pose, 0.0, 0.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)
        
        return policy

    @classmethod
    def _create_stack_policy(cls, action_space: Box,
                             block_size: float) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, _, otherblock = objects
            block_pose = np.array([
                state.get(otherblock, "pose_x"),
                state.get(otherblock, "pose_y"),
                state.get(otherblock, "pose_z")
            ])
            relative_grasp = np.array([
                0.,
                0.,
                block_size,
            ])
            arr = np.r_[block_pose + relative_grasp, 1.0, 0.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_putontable_policy(cls, action_space: Box,
                                  block_size: float) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            x = BlocksEnv.x_lb + (BlocksEnv.x_ub - BlocksEnv.x_lb) * x_norm
            y = BlocksEnv.y_lb + (BlocksEnv.y_ub - BlocksEnv.y_lb) * y_norm
            z = BlocksEnv.table_height + 0.5 * block_size
            arr = np.array([x, y, z, 1.0, 0.0], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy
    
    @classmethod
    def _create_achievegoal_policy(cls, action_space: Box) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params
            top_block, _ = objects
            block_pose = np.array([
                state.get(top_block, "pose_x"),
                state.get(top_block, "pose_y"),
                state.get(top_block, "pose_z")
            ])
            arr = np.r_[block_pose, 1.0, 1.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)
        
        return policy
