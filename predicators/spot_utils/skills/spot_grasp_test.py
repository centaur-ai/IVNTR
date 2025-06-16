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
from predicators.spot_utils.utils import DEFAULT_HAND2STAIR_TF
from predicators.ground_truth_models.spot_view_plan.options import _stable_grasp_at_pixel_and_pickup


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

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
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

        localizer.localize()
        curr_body_pose = localizer.get_last_robot_pose().to_matrix()

        # move hand to indicate stair
        body2hand_rel_pose = math_helpers.SE3Pose(x=0.8, y=0.0, z=-0.2, rot=math_helpers.Quat.from_pitch(np.pi / 2))
        move_hand_to_relative_pose(robot, body2hand_rel_pose)
        time.sleep(1.0)
        localizer.localize()
        curr_hand_pose = localizer.get_last_hand_pose()
        stair_pose = curr_hand_pose * DEFAULT_HAND2STAIR_TF

        input("Press enter when stair is ready")
        stow_arm(robot)
        time.sleep(1.0)
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        rot_thresh = 0.17
        timeout = 20.0
        
        _stable_grasp_at_pixel_and_pickup(robot, localizer, stair_pose, 
                                            top_down_rot, rot_thresh, timeout, False)


    _run_manual_test()
