"""Interface for Spot navigation."""

import logging
import time
import scipy
from typing import Tuple, Collection

from bosdyn.api import manipulation_api_pb2, robot_state_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, \
    get_se2_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.sdk import Robot
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient

from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import get_robot_state, get_spot_home_pose, \
    spot_pose_to_geom2d, _Geom2D, valid_navigation_position, spot_se2pose_to_geom2d
from predicators.utils import Circle, Rectangle
from predicators.spot_utils.planning.rrt import RRT
from predicators.spot_utils.planning.rrt_with_pathsmoothing import path_smoothing

def wait_until_grasp_state_updates(grasp_override_command, robot_state_client):
    updated = False
    has_grasp_override = grasp_override_command.HasField("api_grasp_override")
    has_carry_state_override = grasp_override_command.HasField("carry_state_override")

    while not updated:
        robot_state = robot_state_client.get_robot_state()

        grasp_state_updated = (robot_state.manipulator_state.is_gripper_holding_item and
                               (grasp_override_command.api_grasp_override.override_request
                                == manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING)) or (
                                    not robot_state.manipulator_state.is_gripper_holding_item and
                                    grasp_override_command.api_grasp_override.override_request
                                    == manipulation_api_pb2.ApiGraspOverride.OVERRIDE_NOT_HOLDING)
        carry_state_updated = has_carry_state_override and (
            robot_state.manipulator_state.carry_state
            == grasp_override_command.carry_state_override.override_request)
        updated = (not has_grasp_override or
                   grasp_state_updated) and (not has_carry_state_override or carry_state_updated)
        time.sleep(0.1)


def navigate_to_relative_pose_collision_free(robot: Robot,
                                localizer: SpotLocalizer,
                                body_tform_goal: math_helpers.SE2Pose,
                                collision_geoms: Collection[_Geom2D],
                                allowed_regions: Collection[scipy.spatial.Delaunay],
                                max_xytheta_vel: Tuple[float, float,
                                                    float] = (0.5, 0.5, 0.5),
                                min_xytheta_vel: Tuple[float, float,
                                                    float] = (-0.5, -0.5,
                                                            -0.5),
                                timeout: float = 20.0) -> None:
    localizer.localize()
    robot_pose = localizer.get_last_robot_pose()
    robot_se2 = robot_pose.get_closest_se2_transform()
    tgt_se2 = robot_se2 * body_tform_goal
    curr_robot_geom = spot_pose_to_geom2d(robot_pose)
    tgt_robot_geom = spot_se2pose_to_geom2d(tgt_se2)
    assert valid_navigation_position(curr_robot_geom, collision_geoms, allowed_regions)
    assert valid_navigation_position(tgt_robot_geom, collision_geoms, allowed_regions)
    obstacle_list = []
    for geom in collision_geoms:
        if isinstance(geom, Circle):
            obstacle_list.append((geom.x, geom.y, 0.8))
        elif isinstance(geom, Rectangle):
            x, y, w, h = geom.x, geom.y, geom.width, geom.height
            cx = x + w / 2
            cy = y + h / 2
            obstacle_list.append((cx, cy, 0.8))
        else:
            raise ValueError("Unsupported geom type")
    path_found = False
    while not path_found:
        rrt = RRT(start=[curr_robot_geom.x, curr_robot_geom.y], \
                goal=[tgt_robot_geom.x, tgt_robot_geom.y],
                rand_area=[-4, 5], 
                play_area=[-2.1, 3.8, -3.8, 5.2],
                expand_dis=0.5, max_iter=500,
                obstacle_list=obstacle_list,
                robot_radius=0.4)
        path = rrt.planning(animation=False)
        smoothedPath = path_smoothing(path, 1000, obstacle_list)
        candidate_path = []
        valid = []
        last_x = curr_robot_geom.x
        last_y = curr_robot_geom.y
        for x, y in smoothedPath:
            direction = np.arctan2(y - last_y, x - last_x)
            candidate_se2 = math_helpers.SE2Pose(x=x, y=y, angle=direction)
            candidate_geom = spot_se2pose_to_geom2d(candidate_se2)
            if valid_navigation_position(candidate_geom, collision_geoms, allowed_regions):
                candidate_path.append(candidate_se2)
                valid.append(True)
                last_x = x
                last_y = y
            else:
                break
        if len(candidate_path) == len(smoothedPath):
            path_found = True
    logging.info("Path found")
    for pose in candidate_path:
        navigate_to_absolute_pose_fixhand(robot, localizer, pose, 
                    max_xytheta_vel, min_xytheta_vel, timeout)
        time.sleep(1)


def navigate_to_relative_pose(robot: Robot,
                              body_tform_goal: math_helpers.SE2Pose,
                              max_xytheta_vel: Tuple[float, float,
                                                     float] = (0.5, 0.5, 0.5),
                              min_xytheta_vel: Tuple[float, float,
                                                     float] = (-0.5, -0.5,
                                                               -0.5),
                              timeout: float = 20.0) -> None:
    """Execute a relative move.

    The pose is dx, dy, dyaw relative to the robot's body.
    """
    # Get the robot's current state.
    print("Hello")
    robot_state = get_robot_state(robot)
    transforms = robot_state.kinematic_state.transforms_snapshot
    assert str(transforms) != ""

    # We do not want to command this goal in body frame because the body will
    # move, thus shifting our goal. Instead, we transform this offset to get
    # the goal position in the output frame (odometry).
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME,
                                       BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified
    # frame. The command will stop at the new position.
    # Constrain the robot not to turn, forcing it to strafe laterally.
    speed_limit = SE2VelocityLimit(
        max_vel=SE2Velocity(linear=Vec2(x=max_xytheta_vel[0],
                                        y=max_xytheta_vel[1]),
                            angular=max_xytheta_vel[2]),
        min_vel=SE2Velocity(linear=Vec2(x=min_xytheta_vel[0],
                                        y=min_xytheta_vel[1]),
                            angular=min_xytheta_vel[2]))
    mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)
    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x,
        goal_y=out_tform_goal.y,
        goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME,
        params=mobility_params)
    cmd_id = robot_command_client.robot_command(lease=None,
                                                command=robot_cmd,
                                                end_time_secs=time.time() +
                                                timeout)
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) <= timeout:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.\
            synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:  # pylint: disable=no-member,line-too-long
            logging.warning("Failed to reach the goal")
            return
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL
                and traj_feedback.body_movement_status
                == traj_feedback.BODY_STATUS_SETTLED):
            return
    if (time.perf_counter() - start_time) > timeout:
        logging.warning("Timed out waiting for movement to execute!")

def navigate_to_relative_pose_fixhand(robot: Robot,
                              body_tform_goal: math_helpers.SE2Pose,
                              max_xytheta_vel: Tuple[float, float,
                                                     float] = (0.5, 0.5, 0.5),
                              min_xytheta_vel: Tuple[float, float,
                                                     float] = (-0.5, -0.5,
                                                               -0.5),
                              timeout: float = 20.0) -> None:
    """Execute a relative move.

    The pose is dx, dy, dyaw relative to the robot's body.
    """
    # Get the robot's current state.
    print("Hello")
    robot_state = get_robot_state(robot)
    transforms = robot_state.kinematic_state.transforms_snapshot
    assert str(transforms) != ""

    # We do not want to command this goal in body frame because the body will
    # move, thus shifting our goal. Instead, we transform this offset to get
    # the goal position in the output frame (odometry).
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME,
                                       BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal
    # Overide that the arm is holding something uncarriable
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    grasp_holding_override = manipulation_api_pb2.ApiGraspOverride(
        override_request=manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING)
    not_carriable_override = manipulation_api_pb2.ApiGraspedCarryStateOverride(
        override_request=robot_state_pb2.ManipulatorState.CARRY_STATE_CARRIABLE)
    override_request = manipulation_api_pb2.ApiGraspOverrideRequest(
            api_grasp_override=grasp_holding_override, carry_state_override=not_carriable_override)
    manipulation_client.grasp_override_command(override_request)
    # Wait for the override to take effect before trying to move the arm.
    wait_until_grasp_state_updates(override_request, robot_state_client)

    # Command the robot to go to the goal point in the specified
    # frame. The command will stop at the new position.
    # Constrain the robot not to turn, forcing it to strafe laterally.
    speed_limit = SE2VelocityLimit(
        max_vel=SE2Velocity(linear=Vec2(x=max_xytheta_vel[0],
                                        y=max_xytheta_vel[1]),
                            angular=max_xytheta_vel[2]),
        min_vel=SE2Velocity(linear=Vec2(x=min_xytheta_vel[0],
                                        y=min_xytheta_vel[1]),
                            angular=min_xytheta_vel[2]))
    mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)
    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)
    # default keep arm joint frozen wrt body
    command = RobotCommandBuilder.arm_joint_freeze_command()
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x,
        goal_y=out_tform_goal.y,
        goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME,
        params=mobility_params,
        build_on_command=command)
    cmd_id = robot_command_client.robot_command(lease=None,
                                                command=robot_cmd,
                                                end_time_secs=time.time() +
                                                timeout)
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) <= timeout:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.\
            synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:  # pylint: disable=no-member,line-too-long
            logging.warning("Failed to reach the goal")
            return
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL
                and traj_feedback.body_movement_status
                == traj_feedback.BODY_STATUS_SETTLED):
            return
    if (time.perf_counter() - start_time) > timeout:
        logging.warning("Timed out waiting for movement to execute!")

def navigate_to_absolute_pose(robot: Robot,
                              localizer: SpotLocalizer,
                              target_pose: math_helpers.SE2Pose,
                              max_xytheta_vel: Tuple[float, float,
                                                     float] = (0.5, 0.5, 0.5),
                              min_xytheta_vel: Tuple[float, float,
                                                     float] = (-0.5, -0.5,
                                                               -0.5),
                              timeout: float = 20.0) -> None:
    """Move to the absolute SE2 pose."""
    localizer.localize()
    robot_pose = localizer.get_last_robot_pose()
    robot_se2 = robot_pose.get_closest_se2_transform()
    rel_pose = robot_se2.inverse() * target_pose
    return navigate_to_relative_pose(robot, rel_pose, max_xytheta_vel,
                                     min_xytheta_vel, timeout)

def navigate_to_absolute_pose_fixhand(robot: Robot,
                                localizer: SpotLocalizer,
                                target_pose: math_helpers.SE2Pose,
                                max_xytheta_vel: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                                min_xytheta_vel: Tuple[float, float, float] = (-0.5, -0.5, -0.5),
                                timeout: float = 20.0) -> None:
    """Move to the absolute SE2 pose."""
    localizer.localize()
    robot_pose = localizer.get_last_robot_pose()
    robot_se2 = robot_pose.get_closest_se2_transform()
    rel_pose = robot_se2.inverse() * target_pose
    return navigate_to_relative_pose_fixhand(robot, rel_pose, max_xytheta_vel,
                                        min_xytheta_vel, timeout)

def go_home(robot: Robot,
            localizer: SpotLocalizer,
            max_xytheta_vel: Tuple[float, float, float] = (2.0, 2.0, 1.0),
            min_xytheta_vel: Tuple[float, float, float] = (-2.0, -2.0, -1.0),
            timeout: float = 20.0) -> None:
    """Navigate to a known home position (defined in utils.py)."""
    home_pose = get_spot_home_pose()
    return navigate_to_absolute_pose(robot,
                                     localizer,
                                     home_pose,
                                     max_xytheta_vel=max_xytheta_vel,
                                     min_xytheta_vel=min_xytheta_vel,
                                     timeout=timeout)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # pylint: disable=ungrouped-imports
    import numpy as np
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.settings import CFG
    from predicators.spot_utils.utils import verify_estop, spot_stand, \
        get_graph_nav_dir

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)
        utils.reset_config({
        "spot_graph_nav_map": "debug_place"
        })

        # Get constants.
        hostname = CFG.spot_robot_ip

        sdk = create_standard_sdk('NavigationSkillTestClient')
        robot = sdk.create_robot(hostname)
        path = get_graph_nav_dir()
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(lease_client,
                                         must_acquire=True,
                                         return_at_exit=True)
        robot.time_sync.wait_for_sync()
        spot_stand(robot)
        localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
        robot.time_sync.wait_for_sync()
        relative_poses = [
            # (math_helpers.SE2Pose(x=0, y=0, angle=0), "Standing still"),
            # (math_helpers.SE2Pose(x=0, y=0.5, angle=0), "Moving dy"),
            # (math_helpers.SE2Pose(x=0, y=-0.5, angle=0), "Moving -dy"),
            (math_helpers.SE2Pose(x=0.4, y=-0.0, angle=0.0), "Moving dx"),
            # (math_helpers.SE2Pose(x=-0.4, y=0, angle=0), "Moving -dx"),
            # (math_helpers.SE2Pose(x=0, y=0, angle=np.pi / 2), "Moving yaw"),
            # (math_helpers.SE2Pose(x=0, y=0, angle=-np.pi / 2), "Moving -yaw"),
        ]
        go_home(robot, localizer)
        # navigate_to_relative_pose(robot, math_helpers.SE2Pose(x=0.2, y=0, angle=0))
        # for n in range(10):
        #     localizer_content = {
        #         'onground': {
        #             'body': [],
        #             'hand': []
        #         },
        #         'onstair': {
        #             'body': [],
        #             'hand': []
        #         },
        #     }
        #     for relative_pose, msg in relative_poses:
        #         logging.info(msg)
        #         for _ in range(10):
        #             time.sleep(1)
        #             localizer.localize()
        #             robot_pose = localizer.get_last_robot_pose().to_matrix()
        #             hand_pose = localizer.get_last_hand_pose().to_matrix()
        #             localizer_content['onground']['body'].append(robot_pose)
        #             localizer_content['onground']['hand'].append(hand_pose)
        #             print("Current body pose: %s", localizer.get_last_robot_pose())
        #             print("Current hand pose: %s", localizer.get_last_hand_pose())
        #         navigate_to_relative_pose(robot, relative_pose)
        #         for _ in range(10):
        #             time.sleep(1)
        #             localizer.localize()
        #             robot_pose = localizer.get_last_robot_pose().to_matrix()
        #             hand_pose = localizer.get_last_hand_pose().to_matrix()
        #             localizer_content['onstair']['body'].append(robot_pose)
        #             localizer_content['onstair']['hand'].append(hand_pose)
        #             print("Current body pose: %s", localizer.get_last_robot_pose())
        #             print("Current hand pose: %s", localizer.get_last_hand_pose())
        #     navigate_to_relative_pose(robot, math_helpers.SE2Pose(x=-0.4, y=0, angle=0))
        #     time.sleep(1)
        #     with open(f"on_off_stair_{n}.pkl", "wb") as f:
        #         pickle.dump(localizer_content, f)
    _run_manual_test()
