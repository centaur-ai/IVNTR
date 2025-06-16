"""Ground-truth options for the Spot View Planning environment."""

import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot
from gym.spaces import Box

from predicators import utils
from predicators.envs.spot_env import HANDEMPTY_GRIPPER_THRESHOLD, \
    get_robot, get_robot_gripper_open_percentage
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.spot_utils.kinematics.spot_arm import SpotArmFK
from predicators.spot_utils.perception.spot_cameras import \
    get_last_captured_images, capture_images
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
from predicators.spot_utils.skills.spot_hand_move import close_gripper, \
    gaze_at_relative_pose, move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose_fixhand, navigate_to_relative_pose
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import DEFAULT_HAND_DROP_OBJECT_POSE, \
    DEFAULT_HAND_HOLDING_ANGLES, DEFAULT_HAND_POST_DUMP_POSE, \
    DEFAULT_HAND_PRE_DUMP_LIFT_POSE, DEFAULT_HAND_PRE_DUMP_POSE, \
    get_relative_se2_from_se3, get_relative_se2_from_se3_xy, \
    get_relative_se3_from_distance_and_angles, get_pixel_from_user
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type

###############################################################################
#            Helper functions for chaining multiple spot skills               #
###############################################################################

def _stow_arm_and_move_to_relative_pose(robot: Robot,
                                        rel_pose: math_helpers.SE2Pose) -> None:
    stow_arm(robot)
    navigate_to_relative_pose(robot, rel_pose)
    open_gripper(robot)


def _grasp_at_pixel_and_pickup(
        robot: Robot, localizer: SpotLocalizer, img: RGBDImageWithContext, \
        pixel: Tuple[int, int], grasp_rot: Optional[math_helpers.Quat], rot_thresh: float,
        timeout: float, retry_grasp_after_fail: bool) -> None:
    # Grasp.
    succ = grasp_at_pixel(robot,
                   img,
                   pixel,
                   grasp_rot=grasp_rot,
                   rot_thresh=rot_thresh,
                   timeout=timeout,
                   retry_with_no_constraints=retry_grasp_after_fail)
    # Dump, if the grasp was successful.
    robot_kinematics = SpotArmFK()
    thresh = HANDEMPTY_GRIPPER_THRESHOLD
    if succ:
        localizer.localize()
        body_pose_mat = localizer.get_last_robot_pose().to_matrix()
        desired_hand_pose_mat = robot_kinematics.compute_fk(
                body_pose_mat, DEFAULT_HAND_HOLDING_ANGLES)[-1]
        body_pose_se3 = math_helpers.SE3Pose.from_matrix(body_pose_mat)
        desired_hand_pose_se3 = math_helpers.SE3Pose.from_matrix(desired_hand_pose_mat)
        # Lift the grasped object up high enough that it doesn't collide.
        time.sleep(1.0)
        body_2_hand_pose = body_pose_se3.inverse() * desired_hand_pose_se3
        move_hand_to_relative_pose(robot, body_2_hand_pose)
    # Stow.
    else:
        open_gripper(robot)
        stow_arm(robot)
        back_up_pose = math_helpers.SE2Pose(x=-1.5, y=1.0, angle=0.0)
        navigate_to_relative_pose(robot, back_up_pose)

def _grasp_dummy(robot: Robot) -> None:
    # Dummy grasp function that does nothing.
    pass

def _drop_and_stow(robot: Robot) -> None:
    # First, move the arm to a position from which the object will drop.
    move_hand_to_relative_pose(robot, DEFAULT_HAND_DROP_OBJECT_POSE)
    # Open the hand.
    open_gripper(robot)
    # Stow.
    stow_arm(robot)

def _grasp(robot: Robot) -> None:
    # Calibrate. Use Close Gripper and Open Gripper as indicators.
    close_gripper(robot)
    open_gripper(robot)

def _put(robot: Robot) -> None:
    # Measure Value. Use Close Gripper and Open Gripper Twice as indicators.
    close_gripper(robot)
    open_gripper(robot)
    close_gripper(robot)
    open_gripper(robot)


###############################################################################
#                    Helper parameterized option policies                     #
###############################################################################

def _move_to_target_policy_xy(name: str, dx_param_idx: int,
                        dy_param_idx: int, dyaw_param_idx: int,
                        robot_obj_idx: int, target_obj_idx: int, 
                        stow_arm: bool, state: State,
                           memory: Dict, objects: Sequence[Object],
                           params: Array, param_normalizer: Dict[str, Tuple[float, float]]) \
                            -> Action:

    del memory  # not used

    robot, localizer, _ = get_robot()

    unnormed_dx = params[dx_param_idx]
    unnormed_dy = params[dy_param_idx]
    unnormed_dyaw = params[dyaw_param_idx]
    # assert -1.0 <= unnormed_yaw <= 1.0, f"Yaw parameter needs to be normalized in sampler"
    # assert 0.0 <= unnormed_distance <= 1.0, f"Distance parameter needs to be normalized in sampler"
    
    dx = unnormed_dx * (param_normalizer['dx_max'] - param_normalizer['dx_min']) \
            + param_normalizer['dx_min']
    dy = unnormed_dy * (param_normalizer['dy_max'] - param_normalizer['dy_min']) \
            + param_normalizer['dy_min']
    dyaw = unnormed_dyaw * np.pi
    robot_obj = objects[robot_obj_idx]
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

    target_obj = objects[target_obj_idx]
    target_pose = utils.get_se3_pose_from_state(state, target_obj)

    rel_pose = get_relative_se2_from_se3_xy(robot_pose, target_pose, dx,
                                         dy, dyaw)
    target_height = state.get(target_obj, "height")
    gaze_target = math_helpers.Vec3(target_pose.x, target_pose.y,
                                    target_pose.z + target_height / 2)
    if not stow_arm:
        fn: Callable = navigate_to_relative_pose_fixhand
        fn_args: Tuple = (robot, rel_pose)
    else:
        fn: Callable = _stow_arm_and_move_to_relative_pose
        fn_args: Tuple = (robot, rel_pose)

    return utils.create_spot_env_action(name, objects, fn, fn_args)

def _move_to_on_stair_policy(name: str, state: State, memory: Dict,
                             robot_obj_idx: int, stair_obj_idx: int,
                             objects: Sequence[Object], params: Array) -> Action:
    del memory, params  # not used

    robot, localizer, _ = get_robot()
    rel_pose = math_helpers.SE2Pose(0.4, 0, 0)

    fn: Callable = navigate_to_relative_pose
    fn_args: Tuple = (robot, rel_pose)

    return utils.create_spot_env_action(name, objects, fn, fn_args)

def _move_away_from_policy(name: str, distance_param_idx: int,
                           yaw_param_idx: int, robot_obj_idx: int,
                            target_obj_idx: int, do_gaze: bool, state: State,
                            memory: Dict, objects: Sequence[Object],
                           params: Array, param_normalizer: Dict[str, Tuple[float, float]]) \
                            -> Action:
    del memory  # not used

    robot, localizer, _ = get_robot()

    unnormed_distance = params[distance_param_idx]
    unnormed_yaw = params[yaw_param_idx]
    # assert -1.0 <= unnormed_yaw <= 1.0, f"Yaw parameter needs to be normalized in sampler"
    # assert 0.0 <= unnormed_distance <= 1.0, f"Distance parameter needs to be normalized in sampler"
    # unnormed_distance = np.clip(unnormed_distance, 0.0, 1.0)
    # unnormed_yaw = np.clip(unnormed_yaw, -1.0, 1.0)
    abs_yaw = unnormed_yaw * np.pi
    distance = unnormed_distance * (param_normalizer['distance'][1] - \
            param_normalizer['distance'][0]) + param_normalizer['distance'][0]

    robot_obj = objects[robot_obj_idx]
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

    target_obj = objects[target_obj_idx]
    target_pose = utils.get_se3_pose_from_state(state, target_obj)

    rel_pose = get_relative_se2_from_se3(robot_pose, target_pose, distance,
                                            abs_yaw)
    fn = _stow_arm_and_move_to_relative_pose
    fn_args = (robot, rel_pose)

    return utils.create_spot_env_action(name, objects, fn, fn_args)

def _hand_view_object_policy(name: str, distance_param_idx: int,
                           theta_param_idx: int, phi_param_idx: int, 
                           robot_obj_idx: int,
                           target_obj_idx: int, state: State,
                           memory: Dict, objects: Sequence[Object],
                           params: Array, param_normalizer: Dict[str, Tuple[float, float]]) \
                            -> Action:
    del memory  # not used

    robot, localizer, _ = get_robot()
    robot_obj = objects[robot_obj_idx]
    body_pose = utils.get_se3_pose_from_state(state, robot_obj)

    target_obj = objects[target_obj_idx]
    target_pose = utils.get_se3_pose_from_state(state, target_obj)

    # relative 3D angle
    unnormed_theta = params[theta_param_idx] # angle to z axis
    unnormed_phi = params[phi_param_idx] # angle to x axis
    # assert -1.0 <= unnormed_theta <= 1.0, f"Theta parameter needs to be normalized in sampler"
    # assert -1.0 <= unnormed_phi <= 1.0, f"Phi parameter needs to be normalized in sampler"
    theta = unnormed_theta * np.pi
    phi = unnormed_phi * np.pi

    # distance of the hand
    unnormed_distance = params[distance_param_idx]
    distance = unnormed_distance * (param_normalizer['distance'][1] - \
            param_normalizer['distance'][0]) + param_normalizer['distance'][0]

    body2hand = get_relative_se3_from_distance_and_angles(body_pose, target_pose, 
                        distance, theta, phi)
    
    fn = move_hand_to_relative_pose
    fn_args = (robot, body2hand)

    return utils.create_spot_env_action(name, objects, fn, fn_args)

def _grasp_tgt_policy(name: str, state: State, memory: Dict, objects: Sequence[Object],
                        params: Array) -> Action:
    del state, memory, params  # not used
    robot, localizer, _ = get_robot()
    fn = _grasp
    return utils.create_spot_env_action(name, objects, fn, (robot, ))

def _put_policy(name: str, state: State, memory: Dict, objects: Sequence[Object],
                        params: Array) -> Action:
    del state, memory, params  # not used
    robot, localizer, _ = get_robot()
    fn = _put
    return utils.create_spot_env_action(name, objects, fn, (robot, ))

def _grasp_stair_policy(name: str,
                  target_obj_idx: int,
                  state: State,
                  memory: Dict,
                  objects: Sequence[Object],
                  params: Array,
                  do_dump: bool = False) -> Action:
    del memory, params  # not used

    robot, localizer, _ = get_robot()
    if not CFG.spot_run_dry:
        # Special case: if we're running dry, the image won't be used.
        # default_cam = "frontleft_fisheye_image"
        # rgbd = capture_images(robot, localizer, [default_cam])[default_cam]
        # pixel = get_pixel_from_user(rgbd.rgb)
        # Grasp from the top-down.
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        fn = _grasp_at_pixel_and_pickup

        return utils.create_spot_env_action(
            name, objects, fn, (robot, localizer, top_down_rot, 0.4, 20.0,
                                True))
    else:
        # Dummy grasp function that does nothing.
        fn = _grasp_dummy
        return utils.create_spot_env_action(name, objects, fn, (robot, ))


###############################################################################
#                   Concrete parameterized option policies                    #
###############################################################################


def _move_to_hand_view_object_policy(state: State, memory: Dict,
                                     objects: Sequence[Object],
                                     params: Array) -> Action:
    name = "MoveToHandViewObject"
    dx_param_idx = 0
    dy_param_idx = 1
    dyaw_param_idx = 2
    tgt_yaw = utils.get_se3_pose_from_state(state, objects[1]).rot.to_yaw()
    dx_max = CFG.viewobj_distance[1] * np.cos(tgt_yaw)
    dx_min = CFG.viewobj_distance[0] * np.cos(tgt_yaw)
    dy_max = CFG.viewobj_distance[1] * np.sin(tgt_yaw)
    dy_min = CFG.viewobj_distance[0] * np.sin(tgt_yaw)
    robot_obj_idx = 0
    target_obj_idx = 1
    stow_arm = True
    param_normalizer = {
        'dx_max': max(dx_max, dx_min),
        'dx_min': min(dx_max, dx_min),
        'dy_max': max(dy_max, dy_min),
        'dy_min': min(dy_max, dy_min),
    }
    return _move_to_target_policy_xy(name, dx_param_idx, dy_param_idx,
                                     dyaw_param_idx,
                                  robot_obj_idx, target_obj_idx, stow_arm,
                                  state, memory, objects, params,
                                  param_normalizer)

def _move_away_from_off_stair_policy(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> Action:
    name = "MoveAwayFromOffStair"
    distance_param_idx = 0
    min_distance = CFG.far_distance[0]
    max_distance = CFG.far_distance[1]
    yaw_param_idx = 1
    robot_obj_idx = 0
    target_obj_idx = 1 # the stair
    stow_arm = True
    param_normalizer = {
        'distance': (min_distance, max_distance),
    }
    return _move_away_from_policy(name, distance_param_idx, yaw_param_idx,
                                  robot_obj_idx, target_obj_idx, stow_arm,
                                  state, memory, objects, params,
                                  param_normalizer)

def _move_away_from_on_stair_policy(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> Action:
    name = "MoveAwayFromOnStair"
    distance_param_idx = 0
    min_distance = CFG.far_distance[0]
    max_distance = CFG.far_distance[1]
    yaw_param_idx = 1
    robot_obj_idx = 0
    target_obj_idx = 1 # the stair
    stow_arm = True
    param_normalizer = {
        'distance': (min_distance, max_distance),
    }
    return _move_away_from_policy(name, distance_param_idx, yaw_param_idx,
                                  robot_obj_idx, target_obj_idx, stow_arm,
                                  state, memory, objects, params,
                                  param_normalizer)

def _hand_view_policy(state: State, memory: Dict,
                        objects: Sequence[Object],
                        params: Array) -> Action:
    name = "HandViewObject"
    distance_param_idx = 0
    min_distance = CFG.cam2obj_distance_tol[0]
    max_distance = CFG.cam2obj_distance_tol[1]
    theta_param_idx = 1
    phi_param_idx = 2
    robot_obj_idx = 0
    target_obj_idx = 1
    param_normalizer = {
        'distance': (min_distance, max_distance),
    }
    return _hand_view_object_policy(name, distance_param_idx, theta_param_idx, 
                                    phi_param_idx, robot_obj_idx, target_obj_idx, 
                                    state, memory, objects, params,
                                    param_normalizer)

def _calibrate_obj_policy(state: State, memory: Dict,
                        objects: Sequence[Object],
                        params: Array) -> Action:
    name = "Grasp"
    return _grasp_tgt_policy(name, state, memory, objects, params)

def _measure_obj_policy(state: State, memory: Dict,
                        objects: Sequence[Object],
                        params: Array) -> Action:
    name = "Put"
    return _put_policy(name, state, memory, objects, params)

def _move_to_on_stair_obj_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
    name = "MoveToOnStairsHandViewObject"
    robot_obj_idx = 0
    stair_obj_idx = 1
    return _move_to_on_stair_policy(name, state, memory, 
                                    robot_obj_idx, stair_obj_idx,
                                    objects, params)

def _move_to_reach_object_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
    name = "MoveToReachObject"
    dx_param_idx = 0
    dy_param_idx = 1
    dyaw_param_idx = 2
    robot_obj_idx = 0
    target_obj_idx = 1
    stow_arm = True
    # min_dist = CFG.reach_stair_distance[0]
    tgt_yaw = utils.get_se3_pose_from_state(state, objects[1]).rot.to_yaw()
    dx_max = CFG.reach_stair_distance[1] * np.cos(tgt_yaw)
    dx_min = CFG.reach_stair_distance[0] * np.cos(tgt_yaw)
    dy_max = CFG.reach_stair_distance[1] * np.sin(tgt_yaw)
    dy_min = CFG.reach_stair_distance[0] * np.sin(tgt_yaw)
    param_normalizer = {
        'dx_max': max(dx_max, dx_min),
        'dx_min': min(dx_max, dx_min),
        'dy_max': max(dy_max, dy_min),
        'dy_min': min(dy_max, dy_min),
    }
    return _move_to_target_policy_xy(name, dx_param_idx, dy_param_idx,
                                     dyaw_param_idx,
                                  robot_obj_idx, target_obj_idx, stow_arm,
                                  state, memory, objects, params,
                                  param_normalizer)

def _pick_object_from_top_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
    name = "PickObjectFromTop"
    target_obj_idx = 1
    return _grasp_stair_policy(name, target_obj_idx, state, memory, objects, params)

def _move_to_place_object_policy(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> Action:
    name = "MoveToPlaceStair"
    robot_obj_idx = 0
    target_obj_idx = 2 # note that 1 is the stair

    robot, localizer, _ = get_robot()

    dx_param_idx = 0
    dy_param_idx = 1
    dyaw_param_idx = 2
    stow_arm = False
    tgt_yaw = utils.get_se3_pose_from_state(state, objects[2]).rot.to_yaw()
    # Keep consistent with sampler
    dx_max = abs(CFG.put_stair_tgt_distance * np.cos(tgt_yaw))
    # dx_min = CFG.put_stair_tgt_distance[0] * np.cos(tgt_yaw)
    dy_max = abs(CFG.put_stair_tgt_distance * np.sin(tgt_yaw))

    param_normalizer = {
        'dx_max': dx_max,
        'dx_min': -dx_max,
        'dy_max': dy_max,
        'dy_min': -dy_max,
    }
    return _move_to_target_policy_xy(name, dx_param_idx, dy_param_idx,
                                     dyaw_param_idx,
                                  robot_obj_idx, target_obj_idx, stow_arm,
                                  state, memory, objects, params,
                                  param_normalizer)

def _place_object_in_front_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
    del memory, params, state  # not used

    name = "PlaceObjectInFront"
    robot, localizer, _ = get_robot()
    return utils.create_spot_env_action(name, objects, _drop_and_stow,
                                        (robot, ))

    # # If we're running on the actual robot, we want to be very precise
    # # about the robot's current pose when computing the relative
    # # placement position.
    # if not CFG.spot_run_dry:
    #     assert localizer is not None
    #     localizer.localize()
    #     robot_pose = localizer.get_last_robot_pose()

    # place_rel_pos = robot_pose.inverse() * place_pose
    # return utils.create_spot_env_action(name, objects,
    #                                     _place_at_relative_position_and_stow,
    #                                     (robot, place_rel_pos))


###############################################################################
#                       Parameterized option factory                          #
###############################################################################

_OPERATOR_NAME_TO_PARAM_SPACE = {
    "MoveToReachObject": Box(-np.inf, np.inf, (3, )),  # dx, dy, dyaw
    "MoveToHandViewObject": Box(-np.inf, np.inf, (3, )),  # dx, dy, dyaw
    "MoveToOnStairsHandViewObject": Box(0, 1, (0, )),  # empty
    "MoveToPlaceStair": Box(-np.inf, np.inf, (3, )),  # dx, dy, dyaw
    "MoveAwayFromOnStair": Box(-np.inf, np.inf, (2, )),  # rel dist, dyaw
    "MoveAwayFromOffStair": Box(-np.inf, np.inf, (2, )),  # rel dist, dyaw
    "HandViewObject": Box(-np.inf, np.inf, (3, )),  # rel dist, look at theta/phi
    # x, y pixel in image + quat (qw, qx, qy, qz). If quat is all 0's
    # then grasp is unconstrained
    "PickObjectFromTop": Box(0, 1, (0, )),  # empty
    "Grasp": Box(0, 1, (0, )),  # empty
    "Put": Box(0, 1, (0, )),  # empty
}

# NOTE: the policies MUST be unique because they output actions with extra info
# that includes the name of the operators.
_OPERATOR_NAME_TO_POLICY = {
    "MoveToReachObject": _move_to_reach_object_policy,
    "MoveToHandViewObject": _move_to_hand_view_object_policy,
    "HandViewObject": _hand_view_policy,
    "PickObjectFromTop": _pick_object_from_top_policy,
    "MoveToPlaceStair": _move_to_place_object_policy,
    "MoveToOnStairsHandViewObject": _move_to_on_stair_obj_policy,
    "MoveAwayFromOnStair": _move_away_from_on_stair_policy,
    "MoveAwayFromOffStair": _move_away_from_off_stair_policy,
    "Grasp": _calibrate_obj_policy,
    "Put": _measure_obj_policy,
}


class _SpotParameterizedOption(utils.SingletonParameterizedOption):
    """A parameterized option for spot.

    NOTE: parameterized options MUST be singletons in order to avoid nasty
    issues with the expected atoms monitoring.

    Also note that we need to define the policies outside the class, rather
    than pass the policies into the class, to avoid pickling issues via bosdyn.
    """

    def __init__(self, operator_name: str, types: List[Type]) -> None:
        params_space = _OPERATOR_NAME_TO_PARAM_SPACE[operator_name]
        policy = _OPERATOR_NAME_TO_POLICY[operator_name]
        super().__init__(operator_name, policy, types, params_space)

    def __reduce__(self) -> Tuple:
        return (_SpotParameterizedOption, (self.name, self.types))
    

class PickPlaceGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the Spot View Planning environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pickplace_stair"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot_arm"]
        stair_type = types["stair"]
        obj_type = types["tgt_pickplace"]

        MoveToReach = _SpotParameterizedOption('MoveToReachObject', types=[robot_type, stair_type])
        PickStair = _SpotParameterizedOption('PickObjectFromTop', types=[robot_type, stair_type])
        MoveToPlace = _SpotParameterizedOption('MoveToPlaceStair', types=[robot_type, stair_type, obj_type])
        MoveToOnStairsHandView = _SpotParameterizedOption('MoveToOnStairsHandViewObject', types=[robot_type, stair_type, obj_type])
        MoveToHandView = _SpotParameterizedOption('MoveToHandViewObject', types=[robot_type, obj_type])
        MoveAwayFromOnStair = _SpotParameterizedOption('MoveAwayFromOnStair', types=[robot_type, stair_type, obj_type])
        MoveAwayFromOffStair = _SpotParameterizedOption('MoveAwayFromOffStair', types=[robot_type, stair_type])
        HandView = _SpotParameterizedOption('HandViewObject', types=[robot_type, obj_type])
        Grasp = _SpotParameterizedOption('Grasp', types=[robot_type, obj_type])
        Put = _SpotParameterizedOption('Put', types=[robot_type, obj_type, obj_type])
        
        return {MoveToReach, PickStair, MoveToPlace, MoveToOnStairsHandView,
            MoveToHandView, MoveAwayFromOnStair, MoveAwayFromOffStair, HandView, Grasp, Put}