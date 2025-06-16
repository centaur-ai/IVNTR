"""Interface for spot grasping skill."""

import logging
import time
from typing import Optional, Tuple
from PIL import Image

from bosdyn.api import geometry_pb2, manipulation_api_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, \
    get_vision_tform_body
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.sdk import Robot
from scipy import ndimage
from scipy.spatial import ConvexHull

import pickle
from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext, LanguageObjectDetectionID
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.skills.spot_hand_move import move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.utils import get_robot_state
from predicators.spot_utils.kinematics.spot_arm import SpotArmFK
from predicators.spot_utils.utils import DEFAULT_DUMPED_TF, DEFAULT_HAND_HOLDING_ANGLES


def grasp_at_pixel(robot: Robot,
                   rgbd: RGBDImageWithContext,
                   pixel: Tuple[int, int],
                   grasp_rot: Optional[math_helpers.Quat] = None,
                   rot_thresh: float = 0.17,
                   move_while_grasping: bool = True,
                   timeout: float = 20.0,
                   retry_with_no_constraints: bool = False) -> None:
    """Grasp an object at a specified pixel in the RGBD image, which should be
    from the hand camera and should be up to date with the robot's state.

    The `move_while_grasping` param dictates whether we're allowing the
    robot to automatically move its feet while grasping or not.

    The `retry_with_no_constraints` dictates whether after failing to grasp we
    try again but with all constraints on the grasp removed.
    """
    # assert rgbd.camera_name == "hand_color_image"

    manipulation_client = robot.ensure_client(
        ManipulationApiClient.default_service_name)

    if move_while_grasping:
        # Stow Arm first (only if robot is allowed to move while grasping)
        stow_arm(robot)

    pick_vec = geometry_pb2.Vec2(x=pixel[0], y=pixel[1])

    # Build the proto. Note that the possible settings for walk_gaze_mode
    # can be found here:
    # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html
    walk_gaze_mode = 1 if move_while_grasping else 2
    grasp_params = manipulation_api_pb2.GraspParams(grasp_palm_to_fingertip=0.6)
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=rgbd.transforms_snapshot,
        frame_name_image_sensor=rgbd.frame_name_image_sensor,
        camera_model=rgbd.camera_model,
        walk_gaze_mode=walk_gaze_mode,
        grasp_params=grasp_params)
    # If a desired rotation for the hand was given, add a grasp constraint.
    if grasp_rot is not None:
        robot_state = get_robot_state(robot)
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME  # pylint: disable=no-member
        vision_tform_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot)
        # Rotation from the body to our desired grasp.
        vision_rot = vision_tform_body.rotation * grasp_rot
        # Turn into a proto.
        constraint = grasp.grasp_params.allowable_orientation.add()  # pylint: disable=no-member
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
            vision_rot.to_proto())
        constraint.rotation_with_tolerance.threshold_radians = rot_thresh

    # Create the request.
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp)

    # Send the request.
    cmd_response = manipulation_client.manipulation_api_command(
        manipulation_api_request=grasp_request)

    # Get feedback from the robot and execute grasping, repeating until a
    # proper response is received.
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) <= timeout:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)
        response = manipulation_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)
        if response.current_state in [
                manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION
        ]:
            break
    if (time.perf_counter() - start_time) > timeout:
        logging.warning("Timed out waiting for grasp to execute!")

    # Retry grasping with no constraints if the corresponding arg is true.
    if response.current_state in [
            manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
    ] and retry_with_no_constraints:
        logging.info("WARNING: grasp planning failed, retrying with no constraint")
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=rgbd.transforms_snapshot,
            frame_name_image_sensor=rgbd.frame_name_image_sensor,
            camera_model=rgbd.camera_model,
            walk_gaze_mode=1)
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp)
        cmd_response = manipulation_client.manipulation_api_command(
            manipulation_api_request=grasp_request)
        while (time.perf_counter() - start_time) <= timeout:
            feedback_request = manipulation_api_pb2.\
                ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)
            response = manipulation_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)
            if response.current_state in [
                    manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                    manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                    manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION
            ]:
                break
        if (time.perf_counter() - start_time) > timeout:
            logging.warning("Timed out waiting for grasp to execute!")

    if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
        logging.info("Grasp succeeded! Moving on to Pick Up.")
        return True

if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # pylint: disable=ungrouped-imports
    import numpy as np
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.perception.spot_cameras import capture_images
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.skills.spot_navigation import navigate_to_relative_pose, \
        navigate_to_relative_pose_fixhand
    from predicators.spot_utils.utils import get_graph_nav_dir, \
        get_pixel_from_user, verify_estop, spot_stand

    def _run_manual_test(n, user_input=True) -> None:
        # Put inside a function to avoid variable scoping issues.
        localizer_content = {
            'before_grasp': {
                'body': [],
                'hand': []
            },
            'after_grasp': {
                'body': [],
                'hand': []
            },
            'after_pickup': {
                'body': [],
                'hand': []
            },
            'after_putdown': {
                'body': [],
                'hand': []
            }
        }
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)
        utils.reset_config({
        "spot_graph_nav_map": "sqh_final"
        })

        # Get constants.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()
        sdk = create_standard_sdk('GraspSkillTestClient')
        robot = sdk.create_robot(hostname)
        robot_kinematics = SpotArmFK()
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(lease_client,
                                         must_acquire=True,
                                         return_at_exit=True)
        robot.time_sync.wait_for_sync()
        spot_stand(robot)
        stow_arm(robot)
        localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
        perception_succ = False
        grasping_succ = False

        for _ in range(10):
            # time.sleep(1)
            localizer.localize()
            robot_pose = localizer.get_last_robot_pose().to_matrix()
            hand_pose = localizer.get_last_hand_pose().to_matrix()
            localizer_content['before_grasp']['body'].append(robot_pose)
            localizer_content['before_grasp']['hand'].append(hand_pose)
            print("Current body pose: %s", robot_pose)
            print("Current hand pose: %s", hand_pose)

        # Capture an image.
        # while (not perception_succ) or (not grasping_succ):
            # rand_x = np.random.uniform(-0.1, 0.05)
            # rand_y = np.random.uniform(-0.1, 0.1)
            # rand_angle = np.random.uniform(-np.pi/36, np.pi/36)
            # small_move = math_helpers.SE2Pose(x=rand_x, y=0.0, angle=0.0)
            # navigate_to_relative_pose(robot, small_move)
        rgbds = capture_images(robot, localizer,
                                 camera_names=["frontleft_fisheye_image", "frontright_fisheye_image"])

        # Select a pixel manually.
        if user_input:
            pixel = get_pixel_from_user(rgbd.rgb)
            print(f"Selected pixel: {pixel}")
        else:
            from predicators.spot_utils.perception.object_detection import YOLOSAM, \
                LangSAMModel
            # model = YOLOSAM(
            #     yolo_weights=CFG.yolo_weights,
            #     sam_weights=CFG.sam_weights,
            # )
            # # image_pil = Image.fromarray(rgbd.rotated_rgb).convert("RGB")
            # pixel_infos = model.detect_objects(rgbd)
            model = LangSAMModel()
            pixel_infos = model.segment(rgbds)
            if len(pixel_infos) == 0:
                print("No object detected.")
                return
            else:
                selected_image_name = list(pixel_infos.keys())[0]
                rgbd = rgbds[selected_image_name]
                pixel = pixel_infos[selected_image_name]
            print(f"Selected pixel: {pixel}")

        input("Press enter to grasp at the selected pixel...")
        # Grasp at the pixel with a top-down grasp.
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        grasping_succ = grasp_at_pixel(robot, rgbd, pixel, grasp_rot=top_down_rot)
        if not grasping_succ:
            stow_arm(robot)
            print("Grasp failed.")
            return

        print("Grasp succeeded!")
        for _ in range(10):
            # time.sleep(1)
            localizer.localize()
            body_pose = localizer.get_last_robot_pose().to_matrix()
            hand_pose = localizer.get_last_hand_pose().to_matrix()
            desired_hand_pose = robot_kinematics.compute_fk(
                body_pose, DEFAULT_HAND_HOLDING_ANGLES)[-1]
            body_pose_se3 = math_helpers.SE3Pose.from_matrix(body_pose)
            current_hand_pose_se3 = math_helpers.SE3Pose.from_matrix(hand_pose)
            desired_hand_pose_se3 = math_helpers.SE3Pose.from_matrix(desired_hand_pose)
            localizer_content['after_grasp']['body'].append(body_pose)
            localizer_content['after_grasp']['hand'].append(hand_pose)
            print("Current body pose: %s", body_pose_se3)
            print("Current hand pose: %s", current_hand_pose_se3)
            print("Desired hand pose: %s", desired_hand_pose)

        # wait for user
        input("Press enter when ready to move on")
        body_2_hand_pose = body_pose_se3.inverse() * desired_hand_pose_se3
        move_hand_to_relative_pose(robot, body_2_hand_pose)
        for _ in range(10):
            # time.sleep(1)
            localizer.localize()
            body_pose = localizer.get_last_robot_pose().to_matrix()
            hand_pose = localizer.get_last_hand_pose().to_matrix()
            localizer_content['after_pickup']['body'].append(body_pose)
            localizer_content['after_pickup']['hand'].append(hand_pose)
            print("Current body pose: %s", localizer.get_last_robot_pose())
            print("Current hand pose: %s", localizer.get_last_hand_pose())

        # # wait for user
        input("Press enter when ready to move on")
        body_tform_goal = math_helpers.SE2Pose(x=1.0, y=-2.0, angle=-1.7)
        navigate_to_relative_pose_fixhand(robot, body_tform_goal)

        target_pos = DEFAULT_DUMPED_TF
        input("Press enter when ready to move on")
        move_hand_to_relative_pose(robot, target_pos)
        for _ in range(10):
            # time.sleep(1)
            localizer.localize()
            body_pose = localizer.get_last_robot_pose().to_matrix()
            hand_pose = localizer.get_last_hand_pose().to_matrix()
            localizer_content['after_putdown']['body'].append(body_pose)
            localizer_content['after_putdown']['hand'].append(hand_pose)
            print("Current body pose: %s", localizer.get_last_robot_pose())
            print("Current hand pose: %s", localizer.get_last_hand_pose())
        
        # wait for user
        input("Press enter when ready to move on")
        open_gripper(robot)
        time.sleep(1.0)
        stow_arm(robot)
        input("Press enter when ready to move on")
        body_tform_goal = math_helpers.SE2Pose(x=0.4, y=0.0, angle=0.0)
        navigate_to_relative_pose(robot, body_tform_goal)

        # input("Press enter when ready to move on")
        # target_pos = math_helpers.SE3Pose(x=0.8, y=0.0, z=0.6, rot=math_helpers.Quat.from_pitch(0.7))
        # input("Press enter when ready to move on")
        # move_hand_to_relative_pose(robot, target_pos)

        # Save the localizer content.
        with open(f"pickup_place_{n}.pkl", "wb") as f:
            pickle.dump(localizer_content, f)

    _run_manual_test(7, False)
