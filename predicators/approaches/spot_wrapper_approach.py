"""An approach that wraps a base approach for a spot environment. Detects when
objects have been "lost" and executes a specific controller to "find" them.
Passes control back to the main approach once all objects are not lost.

Assumes that some objects in the environment have a feature called "lost" that
is 1.0 if the object is lost and 0.0 otherwise. This feature should be tracked
by a perceiver.

For now, the "find" policy is represented with a single action that is
extracted from the environment.
"""
import time
import logging
from typing import Any, Callable, List, Optional, Set

from gym.spaces import Box
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators import utils
from predicators.settings import CFG
from predicators.approaches import BaseApproach, BaseApproachWrapper
from predicators.envs.spot_env import get_detection_id_for_object, get_robot
from predicators.spot_utils.skills.spot_find_objects import find_objects
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.utils import get_allowed_map_regions
from predicators.structs import Action, Object, ParameterizedOption, \
    Predicate, State, Task, Type
from predicators.spot_utils.skills.spot_navigation import navigate_to_relative_pose
from predicators.spot_utils.skills.spot_hand_move import open_gripper

def _stow_arm_and_move_to_relative_pose(robot: Robot,
                        rel_pose: math_helpers.SE2Pose) -> None:
    stow_arm(robot)
    time.sleep(0.5)
    navigate_to_relative_pose(robot, rel_pose)
    open_gripper(robot)

class SpotWrapperApproach(BaseApproachWrapper):
    """Always "find" if some object is lost."""

    def __init__(self, base_approach: BaseApproach,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(base_approach, initial_predicates, initial_options,
                         types, action_space, train_tasks)
        self._base_approach_has_control = False  # for execution monitoring
        self._allowed_regions = get_allowed_map_regions()

    @classmethod
    def get_name(cls) -> str:
        return "spot_wrapper"

    @property
    def is_learning_based(self) -> bool:
        return self._base_approach.is_learning_based

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        # Maintain policy from the base approach.
        base_approach_policy: Optional[Callable[[State], Action]] = None
        self._base_approach_has_control = False
        need_stow = False

        def _policy(state: State) -> Action:
            nonlocal base_approach_policy, need_stow
            # If we think that we're done, return the done action.
            if task.goal_holds(state):
                return utils.create_spot_env_action("done")
            # If some objects are lost, find them.
            lost_objects: Set[Object] = set()
            for obj in state:
                if "lost" in obj.type.feature_names and \
                    state.get(obj, "lost") > 0.5:
                    lost_objects.add(obj)
            # Need to find the objects.
            if lost_objects:
                logging.info(f"[Spot Wrapper] Lost objects: {lost_objects}")
                # Reset the base approach policy.
                base_approach_policy = None
                need_stow = True
                self._base_approach_has_control = False
                robot, localizer, lease_client = get_robot()
                lost_object_ids = {
                    get_detection_id_for_object(o)
                    for o in lost_objects
                }
                allowed_regions = self._allowed_regions
                return utils.create_spot_env_action(
                    "find-objects", [], find_objects,
                    (state, self._rng, robot, localizer, lease_client,
                     lost_object_ids, allowed_regions))
            # Found the objects. Stow the arm before replanning.
            if need_stow:
                logging.info("[Spot Wrapper] Lost objects found, stowing.")
                base_approach_policy = None
                need_stow = False
                self._base_approach_has_control = False
                robot, _, _ = get_robot()
                return utils.create_spot_env_action("stow-arm", [], stow_arm,
                                                    (robot, ))
            # Check if we need to re-solve.
            if base_approach_policy is None:
                logging.info("[Spot Wrapper] Replanning with base approach.")
                cur_task = Task(state, task.goal)
                base_approach_policy = self._base_approach.solve(
                    cur_task, timeout)
                self._base_approach_has_control = True
                # Need to call this once here to fix off-by-one issue.
                # atom_seq = self._base_approach.get_execution_monitoring_info()
                # assert all(a.holds(state) for a in atom_seq[0])
            # Augment the base policy. Some of the options requires external perception info
            return base_approach_policy(state)
            # return self.augment(state, base_approach_policy(state))

        return _policy

    def get_execution_monitoring_info(self) -> List[Any]:
        if self._base_approach_has_control:
            return self._base_approach.get_execution_monitoring_info()
        return []
    
    # def augment(self, 
    #             state: State,
    #             action: Action) -> Action:
    #     """Augment the action with the "spot_wrapper" approach."""
    #     action_name = action.extra_info[0]
    #     if ("Pick" not in action_name) or (CFG.spot_run_dry):
    #         return action
    #     # Augment the action with data from perception states
    #     original_fn_args = action.extra_info[3]
    #     robot, localizer, top_down_rot, perc, degree, retry = \
    #         original_fn_args
    #     if "rgbd" not in state.augmented_state.keys():
    #         logging.warning("perception failed for pick, will move back and replan")
    #         action_name = "HackMoveBack"
    #         objects = [action.extra_info[1][0]] # spot
    #         fn = _stow_arm_and_move_to_relative_pose
    #         rel_pose = math_helpers.SE2Pose(x=-1.2, y=0.5, angle=0)
    #         fn_args = (robot, rel_pose)
    #         new_action = utils.create_spot_env_action(action_name, objects, fn, fn_args)
    #         return new_action
    #     rgbd, pixels = state.augmented_state["rgbd"], state.augmented_state["pixels"]
    #     new_action = utils.create_spot_env_action(
    #         action_name, action.extra_info[1], action.extra_info[2], \
    #         (robot, localizer, rgbd, pixels, top_down_rot, perc, degree,
    #                             retry))
    #     return new_action