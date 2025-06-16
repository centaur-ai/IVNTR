"""Small utility functions for spot."""

import functools
import sys
import logging
from enum import Enum
from pathlib import Path
from typing import Collection, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy
import yaml
from scipy.spatial.transform import Rotation as R
from bosdyn.api import estop_pb2, robot_state_pb2
from bosdyn.client import math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.exceptions import ProxyConnectionError, TimedOutError
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.sdk import Robot
from numpy.typing import NDArray
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module
from predicators.spot_utils.kinematics.spot_arm import SpotArmFK

from predicators import utils
from predicators.settings import CFG
from predicators.structs import Object, State, Type
from predicators.utils import Rectangle, _Geom2D

# Pose for the hand (relative to the body) that looks down in front.
DEFAULT_HAND_LOOK_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 6))
DEFAULT_HAND_DROP_OBJECT_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=-0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2))
DEFAULT_HAND_LOOK_FLOOR_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 3))
DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2))
DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE_HIGH = math_helpers.SE3Pose(
    x=0.65, y=0.0, z=0.32, rot=math_helpers.Quat.from_pitch(np.pi / 2.5))
DEFAULT_HAND_PRE_DUMP_LIFT_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.3, rot=math_helpers.Quat.from_pitch(2 * np.pi / 3))
DEFAULT_HAND_PRE_DUMP_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(-np.pi / 6))
DEFAULT_HAND_POST_DUMP_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2))

# For simulated
DEFAULT_HAND_STOW_ANGLES = [1.03712082e-04, -3.11518478e+00, 3.13274956e+00, 1.57154214e+00, -1.90141201e-02, -1.57168961e+00]
DEFAULT_HAND_HOLDING_ANGLES = [0.1558527158670593, -1.0679405155353672, 1.3354094948097246, 0.002664515178258268, 1.3207200093828653, 0.17407994994706474]
# Pose for the body (relative to the foot).
rot_mat_stair = np.array([[-0.93375213,0.03086331,0.35658718], [-0.02810338,-0.99952153,0.01291956], [0.3568153,0.00204236,0.93417272]])
DEFAULT_STAIR2BODY_ONSTAIR_TF = math_helpers.SE3Pose(
    x=0.542, y=0.0, z=0.343, rot=math_helpers.Quat.from_matrix(rot_mat_stair))
# Pose for the object dumped (relative to the body).
DEFAULT_DUMPED_TF = math_helpers.SE3Pose(
    x=0.8, y=0.0, z=-0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2.5))
# on real spot, we use np.pi / 2, otherwise the error in "z" will cause planning to fail
DEFAULT_DUMPED_TF_REAL = math_helpers.SE3Pose(
    x=0.8, y=0.0, z=-0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2))
# Pose for the hand to pick up stair, relative to the stair (only used in task init)
rot_mat = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
DEFAULT_STAIR2HAND_TF = math_helpers.SE3Pose(
    x=0.0, y=0.0, z=0.0, rot=math_helpers.Quat.from_matrix(rot_mat))
DEFAULT_HAND2STAIR_TF = DEFAULT_STAIR2HAND_TF.inverse()
# Pose for the hand to pick up object, relative to the object (only used in task init)
rot_mat_obj = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
DEFAULT_OBJ2HAND_TF = math_helpers.SE3Pose(
    x=0.0, y=0.0, z=0.0, rot=math_helpers.Quat.from_matrix(rot_mat_obj))
DEFAULT_HAND2OBJ_TF = DEFAULT_OBJ2HAND_TF.inverse()

# Spot-specific types.
class _Spot3DShape(Enum):
    """Stored as an object 'shape' feature."""
    CUBOID = 1
    CYLINDER = 2


_robot_type = Type(
    "robot",
    ["gripper_open_percentage", "x", "y", "z", "qw", "qx", "qy", "qz"])
_robot_hand_type = Type(
    "robot_arm",
    list(_robot_type.feature_names) + 
    ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"] +
    ["calibration_obj_id", "calibrated"],
    parent=_robot_type)
# NOTE: include a unique object identifier in the object state to allow for
# object-specific sampler learning (e.g., pick hammer vs pick brush).
_base_object_type = Type("base-object", [
    "x",
    "y",
    "z",
    "qw",
    "qx",
    "qy",
    "qz",
    "shape",
    "height",
    "width",
    "length",
    "object_id",
])
_movable_object_type = Type(
    "movable",
    list(_base_object_type.feature_names) +
    ["placeable", "held", "lost", "in_hand_view", "in_view", "is_sweeper"],
    parent=_base_object_type)
_immovable_object_type = Type("immovable",
                              list(_base_object_type.feature_names) +
                              ["flat_top_surface"],
                              parent=_base_object_type)
# a target object need to be measured by the robot
_target_object_type = Type("target",
    list(_base_object_type.feature_names) +
    ["measured"] +
    ["r", "g", "b"], # for visualization
    parent=_base_object_type)
# a stair object than can be stand on to measure the other target
_stair_object_type = Type("stair",
    list(_base_object_type.feature_names) +
    ["held", "lost", "in_hand_view", "in_view"] +
    ["r", "g", "b"], # for visualization
    parent=_base_object_type)
_target_object_w_stair_type = Type("target",
    list(_base_object_type.feature_names) +
    ["lost", "in_hand_view", "in_view"] +
    ["measured"] +
    ["stair_id"] +
    ["r", "g", "b"], # for visualization
    parent=_base_object_type)
_target_pickplace_type = Type("tgt_pickplace",
    list(_base_object_type.feature_names) +
    ["held", "lost", "in_hand_view", "in_view"] +
    ["achieved"] +
    ["stair_id"] +
    ["r", "g", "b"], # for visualization
    parent=_base_object_type)

_container_type = Type("container",
                       list(_movable_object_type.feature_names),
                       parent=_movable_object_type)


def get_collision_geoms_for_nav(state: State) -> List[_Geom2D]:
    """Get all relevant collision geometries for navigating."""
    # We want to consider collisions with all objects that:
    # (1) aren't the robot
    # (2) aren't in an excluded object list defined below
    # (3) aren't being currently held.
    # (4) aren't stairs
    excluded_objects = ["robot", "floor", "brush", "yogurt", "football", "spot"]
    collision_geoms = []
    for obj in set(state):
        if obj.name not in excluded_objects:
            if obj.type == _movable_object_type:
                if state.get(obj, "held") > 0.5:
                    continue
            if obj.type == _stair_object_type:
                continue
            x = state.get(obj, "x")
            y = state.get(obj, "y")
            obj_geo = utils.Circle(x, y, CFG.min_distance2tgt_anytime)
            collision_geoms.append(obj_geo)
    return collision_geoms

def get_close_geoms_for_nav(state: State) -> List[_Geom2D]:
    """Get all relevant viewable geometries for navigating."""
    viewable_geoms = []
    # important, consider both target and stairs
    for obj in set(state):
        if obj.type == _stair_object_type:
            x = state.get(obj, "x")
            y = state.get(obj, "y")
            viewable_region = utils.Circle(x, y, CFG.far_distance[0])
            viewable_geoms.append(viewable_region)
    for obj in set(state):
        if obj.type == _target_object_w_stair_type:
            x = state.get(obj, "x")
            y = state.get(obj, "y")
            viewable_region = utils.Circle(x, y, CFG.far_distance[0])
            viewable_geoms.append(viewable_region)
    for obj in set(state):
        if obj.type == _target_pickplace_type:
            if state.get(obj, "held") > 0.5:
                # don't consider the object in hand
                continue
            x = state.get(obj, "x")
            y = state.get(obj, "y")
            viewable_region = utils.Circle(x, y, CFG.far_distance[0])
            viewable_geoms.append(viewable_region)
    return viewable_geoms


def object_to_top_down_geom(
        obj: Object,
        state: State,
        size_buffer: float = 0.0,
        put_on_robot_if_held: bool = True) -> utils._Geom2D:
    """Convert object to top-down view geometry."""
    assert obj.is_instance(_base_object_type)
    shape_type = int(np.round(state.get(obj, "shape")))
    if put_on_robot_if_held and \
        obj.is_instance(_movable_object_type) and state.get(obj, "held") > 0.5:
        robot, = state.get_objects(_robot_type)
        se3_pose = utils.get_se3_pose_from_state(state, robot)
    else:
        se3_pose = utils.get_se3_pose_from_state(state, obj)
    angle = se3_pose.rot.to_yaw()
    center_x = se3_pose.x
    center_y = se3_pose.y
    width = state.get(obj, "width") + size_buffer
    length = state.get(obj, "length") + size_buffer
    if shape_type == _Spot3DShape.CUBOID.value:
        return utils.Rectangle.from_center(center_x, center_y, width, length,
                                           angle)
    assert shape_type == _Spot3DShape.CYLINDER.value
    assert np.isclose(width, length)
    radius = width / 2
    return utils.Circle(center_x, center_y, radius)


def object_to_side_view_geom(
        obj: Object,
        state: State,
        size_buffer: float = 0.0,
        put_on_robot_if_held: bool = True) -> utils._Geom2D:
    """Convert object to side view geometry."""
    assert obj.is_instance(_base_object_type)
    # The shape doesn't matter because all shapes are rectangles from the side.
    # If the object is held, use the robot's pose.
    if put_on_robot_if_held and \
        obj.is_instance(_movable_object_type) and state.get(obj, "held") > 0.5:
        robot, = state.get_objects(_robot_type)
        se3_pose = utils.get_se3_pose_from_state(state, robot)
    else:
        se3_pose = utils.get_se3_pose_from_state(state, obj)
    center_y = se3_pose.y
    center_z = se3_pose.z
    length = state.get(obj, "length") + size_buffer
    height = state.get(obj, "height") + size_buffer
    return utils.Rectangle.from_center(center_y, center_z, length, height, 0.0)


@functools.lru_cache(maxsize=None)
def get_allowed_map_regions() -> Collection[Delaunay]:
    """Gets Delaunay regions from metadata that correspond to free space."""
    metadata = load_spot_metadata()
    allowed_regions = metadata.get("allowed-regions", {})
    convex_hulls = []
    for region_pts in allowed_regions.values():
        dealunay_hull = Delaunay(np.array(region_pts))
        convex_hulls.append(dealunay_hull)
    return convex_hulls

@functools.lru_cache(maxsize=None)
def get_allowed_stair_regions() -> Collection[Delaunay]:
    """Gets Delaunay regions from metadata that correspond to free space."""
    metadata = load_spot_metadata()
    allowed_regions = metadata.get("stair-regions", {})
    convex_hulls = []
    for region_pts in allowed_regions.values():
        dealunay_hull = Delaunay(np.array(region_pts))
        convex_hulls.append(dealunay_hull)
    return convex_hulls

@functools.lru_cache(maxsize=None)
def get_allowed_obj_regions() -> Collection[Delaunay]:
    """Gets Delaunay regions from metadata that correspond to free space."""
    metadata = load_spot_metadata()
    allowed_regions = metadata.get("obj-regions", {})
    convex_hulls = []
    for region_pts in allowed_regions.values():
        dealunay_hull = Delaunay(np.array(region_pts))
        convex_hulls.append(dealunay_hull)
    return convex_hulls

def get_graph_nav_dir() -> Path:
    """Get the path to the graph nav directory."""
    upload_dir = Path(__file__).parent / "graph_nav_maps"
    return upload_dir / CFG.spot_graph_nav_map


def load_spot_metadata() -> Dict:
    """Load from the YAML config."""
    config_filepath = get_graph_nav_dir() / "metadata.yaml"
    with open(config_filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_spot_home_pose() -> math_helpers.SE2Pose:
    """Load the home pose for the robot."""
    metadata = load_spot_metadata()
    home_pose_dict = metadata["spot-home-pose"]
    x = home_pose_dict["x"]
    y = home_pose_dict["y"]
    angle = home_pose_dict["angle"]
    return math_helpers.SE2Pose(x, y, angle)


def get_april_tag_transform(april_tag: int) -> math_helpers.SE3Pose:
    """Load the world frame transform for an april tag.

    Returns identity if no config is found.
    """
    metadata = load_spot_metadata()
    transform_dict = metadata["april-tag-offsets"]
    try:
        april_tag_transform_dict = transform_dict[f"tag-{april_tag}"]
    except KeyError:
        return math_helpers.SE3Pose(0, 0, 0, rot=math_helpers.Quat())
    x = april_tag_transform_dict["x"]
    y = april_tag_transform_dict["y"]
    z = april_tag_transform_dict["z"]
    return math_helpers.SE3Pose(x, y, z, rot=math_helpers.Quat())


def verify_estop(robot: Robot) -> None:
    """Verify the robot is not estopped."""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external" + \
            " E-Stop client, such as the estop SDK example, to" + \
            " configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)


def get_pixel_from_user(rgb: NDArray[np.uint8]) -> Tuple[int, int]:
    """Use open CV GUI to select a pixel on the given image."""

    image_click: Optional[Tuple[int, int]] = None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _callback(event: int, x: int, y: int, flags: int, param: None) -> None:
        """Callback for the click-to-grasp functionality with the Spot API's
        grasping interface."""
        del flags, param
        nonlocal image_click
        if event == cv2.EVENT_LBUTTONUP:
            image_click = (x, y)

    image_title = "Click to grasp"
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, _callback)
    cv2.imshow(image_title, bgr)

    while image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            # Quit and terminate the process (if you're panicking.)
            sys.exit()

    cv2.destroyAllWindows()

    return image_click


def get_relative_se2_from_se3(
        robot_pose: math_helpers.SE3Pose,
        target_pose: math_helpers.SE3Pose,
        target_offset_distance: float = 0.0,
        abs_angle: float = 0.0) -> math_helpers.SE2Pose:
    """Given a current se3 pose and a target se3 pose on the same plane, return
    a relative se2 pose for moving from the current to the target.

    Also add an angle and distance offset to the target pose. The returned
    se2 pose is facing toward the target.

    Typical use case: we know the current se3 pose for the body of the robot
    and the se3 pose for a table, and we want to move in front of the table.
    """
    dx = np.cos(abs_angle) * target_offset_distance
    dy = np.sin(abs_angle) * target_offset_distance
    x = target_pose.x + dx
    y = target_pose.y + dy
    # Face towards the target.
    rot = abs_angle + np.pi if abs_angle < 0 else abs_angle - np.pi
    target_se2 = math_helpers.SE2Pose(x, y, rot)
    robot_se2 = robot_pose.get_closest_se2_transform()
    return robot_se2.inverse() * target_se2


def get_relative_se2_from_se3_xy(
        robot_pose: math_helpers.SE3Pose,
        target_pose: math_helpers.SE3Pose,
        target_offset_dx: float = 0.0,
        target_offset_dy: float = 0.0,
        target_offset_dangle: float = 0.0) -> math_helpers.SE2Pose:
    """Given a current se3 pose and a target se3 pose on the same plane, return
    a relative se2 pose for moving from the current to the target.

    Also add an angle and distance offset to the target pose. The returned
    se2 pose is facing toward the target.

    Typical use case: we know the current se3 pose for the body of the robot
    and the se3 pose for a table, and we want to move in front of the table.
    """
    x = target_pose.x + target_offset_dx
    y = target_pose.y + target_offset_dy
    # Face towards the target.
    rot = target_pose.rot.to_yaw() + target_offset_dangle
    target_se2 = math_helpers.SE2Pose(x, y, rot)
    robot_se2 = robot_pose.get_closest_se2_transform()
    return robot_se2.inverse() * target_se2

def valid_navigation_position(
        robot_geom: Rectangle,
        collision_geoms: Collection[_Geom2D],
        allowed_regions: Collection[scipy.spatial.Delaunay],  # pylint: disable=no-member
) -> bool:
    """Checks whether a given robot geom is not in collision and also within
    some allowed region."""
    # Check for out-of-bounds. To do this, we're looking for
    # one allowed region where all four points defining the
    # robot will be within the region in the new pose.
    oob = True
    for region in allowed_regions:
        for cx, cy in robot_geom.vertices:
            if region.find_simplex(np.array([cx, cy])) < 0:
                break
        else:
            oob = False
            break
    if oob:
        return False
    # Check for collisions.
    collision = False
    for collision_geom in collision_geoms:
        if collision_geom.intersects(robot_geom):
            collision = True
            break
    # Success!
    return not collision


def sample_random_nearby_point_to_move(
    robot_geom: Rectangle,
    collision_geoms: Collection[_Geom2D],
    rng: np.random.Generator,
    max_distance_away: float,
    allowed_regions: Collection[scipy.spatial.Delaunay],  # pylint: disable=no-member
    max_samples: int = 1000
) -> Tuple[float, float, Rectangle]:
    """Sampler for navigating to a randomly selected point within some distance
    from the current robot's position. Useful when trying to find lost objects.

    Returns a distance and an angle in radians. Also returns the next
    robot geom for visualization and debugging convenience.
    """
    for _ in range(max_samples):
        distance = rng.uniform(0.1, max_distance_away)
        angle = rng.uniform(-np.pi, np.pi)
        dx = np.cos(angle) * distance
        dy = np.sin(angle) * distance
        x = robot_geom.x + dx
        y = robot_geom.y + dy
        # Face towards the target.
        rot = angle + np.pi if angle < 0 else angle - np.pi
        cand_geom = Rectangle.from_center(x, y, robot_geom.width,
                                          robot_geom.height, rot)
        if valid_navigation_position(cand_geom, collision_geoms,
                                     allowed_regions):
            return (distance, angle, cand_geom)

    raise RuntimeError(f"Sampling failed after {max_samples} attempts")

def sample_move_offset_with_ik(
    target_pose: math_helpers.SE3Pose,
    body_height: float,
    robot_geom: Rectangle,
    collision_geoms: Collection[_Geom2D],
    rng: np.random.Generator,
    dx: List[float],
    dy: List[float],
    allowed_regions: Collection[scipy.spatial.Delaunay],  # pylint: disable=no-member
    objective: str,
    stair_height: float = 0.0,
    max_samples: int = 1000,
    dtheta: float = np.pi,
    spot_running: bool = False
) -> Tuple[float, float, float, Rectangle]:
    tgt_x = target_pose.x
    tgt_y = target_pose.y
    tgt_theta = target_pose.rot.to_yaw()
    pointing_theta = tgt_theta + np.pi if tgt_theta < 0 \
        else tgt_theta - np.pi
    body_constraint = [
        [tgt_x + min(dx), tgt_x + max(dx)],
        [tgt_y + min(dy), tgt_y + max(dy)],
        [body_height, body_height + 1e-5],
        [0, 1e-5],
        [0, 1e-5],
        [pointing_theta - dtheta, pointing_theta + dtheta],
    ]
    # compute hand pose based on different objectives
    i = 0
    if objective == "handsee":
        tgt_x_axis = target_pose.to_matrix()[:3, 0]
        tgt_z_axis = target_pose.to_matrix()[:3, 2]
        tgt_position = np.array([target_pose.x, 
                                target_pose.y,
                                target_pose.z])
        hand_tgt_dist = (CFG.cam2obj_distance_tol_sampler[0] + \
                         CFG.cam2obj_distance_tol_sampler[1]) / 2
        hand_position = tgt_position + hand_tgt_dist * tgt_x_axis
        hand_x_axis = -tgt_x_axis
        hand_z_axis = tgt_z_axis
        hand_y_axis = np.cross(hand_z_axis, hand_x_axis)
        hand_rot = math_helpers.Quat.from_matrix(np.array([hand_x_axis, hand_y_axis, hand_z_axis]).T)
        hand_pose = math_helpers.SE3Pose(x=hand_position[0], y=hand_position[1], z=hand_position[2], rot=hand_rot)
        hand_pose_mat = hand_pose.to_matrix()
    elif objective == "grasp":
        # Check if the robot can grasp the object after moving.
        # hand pose
        hand_pose = target_pose.mult(DEFAULT_STAIR2HAND_TF)
        hand_pose_mat = hand_pose.to_matrix()
    elif objective == "put_stair":
        tgt_x_axis = target_pose.to_matrix()[:3, 0]
        tgt_z_axis = target_pose.to_matrix()[:3, 2]
        tgt_position = np.array([target_pose.x, 
                                target_pose.y,
                                target_pose.z])
        hand_tgt_dist = (CFG.cam2obj_distance_tol_sampler[0] + \
                         CFG.cam2obj_distance_tol_sampler[1]) / 2
        hand_position = tgt_position + hand_tgt_dist * tgt_x_axis
        hand_x_axis = -tgt_x_axis
        hand_z_axis = tgt_z_axis
        hand_y_axis = np.cross(hand_z_axis, hand_x_axis)
        hand_rot = math_helpers.Quat.from_matrix(np.array([hand_x_axis, hand_y_axis, hand_z_axis]).T)
        hand_pose = math_helpers.SE3Pose(x=hand_position[0], y=hand_position[1], z=hand_position[2], rot=hand_rot)
        hand_pose_mat = hand_pose.to_matrix()
        assert stair_height != 0.0
        # need to change the body constraint to include the stair
        # calculate the spot pitch and z if on stair
        body_pitch_on_stair = DEFAULT_STAIR2BODY_ONSTAIR_TF.rot.to_pitch()
        # Note that stair_height includes the ground z
        stair_z = stair_height + CFG.viewplan_ground_z
        imagined_stair_pose = math_helpers.SE3Pose(x=0, y=0, z=stair_z, 
                                                   rot=math_helpers.Quat())
        imagined_onstair = imagined_stair_pose.mult(DEFAULT_STAIR2BODY_ONSTAIR_TF)
        body_height_on_stair = imagined_onstair.z
        # different from the body height (on ground)
        body_constraint[2] = [body_height_on_stair - 1e-5, body_height_on_stair + 1e-5]
        # different from the body pitch (on ground)
        body_constraint[4] = [body_pitch_on_stair - 1e-5, body_pitch_on_stair + 1e-5]
    else:
        raise ValueError(f"Objective {objective} not supported")

    for _ in range(max_samples):
        i += 1
        logging.info(f"Sampling {i} times")
        init_body_x = rng.uniform(body_constraint[0][0], body_constraint[0][1])
        init_body_y = rng.uniform(body_constraint[1][0], body_constraint[1][1])
        init_body_theta = rng.uniform(body_constraint[5][0], body_constraint[5][1])
        init_body_vec = [init_body_x, init_body_y, body_height, \
                        0, 0, init_body_theta]
        spot_arm_fk = SpotArmFK()
        suc, sol = spot_arm_fk.compute_whole_body_ik(hand_pose_mat, body_constraint, init_body_vec)
        if suc:
            x, y, z, _, _, pointing_theta = sol[:6]
            cand_geom = Rectangle.from_center(x, y, robot_geom.width,
                                        robot_geom.height, pointing_theta)
            if valid_navigation_position(cand_geom, collision_geoms,
                                allowed_regions):
                if objective == "put_stair":
                    # need to transform to body pose off stair
                    body_pose_xyz = sol[:3]
                    body_rot_angle = sol[3:6]
                    body_rot_on_stair_mat = R.from_euler('xyz', body_rot_angle).as_matrix()
                    body_pose = math_helpers.SE3Pose(x=body_pose_xyz[0], 
                                y=body_pose_xyz[1], z=body_pose_xyz[2],
                                rot=math_helpers.Quat.from_matrix(body_rot_on_stair_mat))
                    # compute foot pose, as the stair pose
                    stair_pose = body_pose.mult(DEFAULT_STAIR2BODY_ONSTAIR_TF.inverse())
                    hand_grasping_pose = stair_pose.mult(DEFAULT_STAIR2HAND_TF)
                    if spot_running:
                        # use np.pi / 2, otherwise the error in "z" will cause planning to fail
                        body_pose_off_stair = hand_grasping_pose.mult(DEFAULT_DUMPED_TF_REAL.inverse())
                    else:
                        body_pose_off_stair = hand_grasping_pose.mult(DEFAULT_DUMPED_TF.inverse())
                    x = body_pose_off_stair.x
                    y = body_pose_off_stair.y
                    pointing_theta = body_pose_off_stair.rot.to_yaw()
                    cand_geom_off = Rectangle.from_center(x, y, robot_geom.width,
                                                robot_geom.height, pointing_theta)
                    # assume if on stair ok, off stair is also ok
                    # if valid_navigation_position(cand_geom_off, collision_geoms,
                    #             allowed_regions):
                    dx = x - tgt_x
                    dy = y - tgt_y
                    dtheta = pointing_theta - tgt_theta
                    if dtheta > np.pi:
                        dtheta -= 2 * np.pi
                    if dtheta < -np.pi:
                        dtheta += 2 * np.pi
                    return (dx, dy, dtheta, cand_geom_off)
                else:
                    dx = x - tgt_x
                    dy = y - tgt_y
                    dtheta = pointing_theta - tgt_theta
                    if dtheta > np.pi:
                        dtheta -= 2 * np.pi
                    if dtheta < -np.pi:
                        dtheta += 2 * np.pi
                return (dx, dy, dtheta, cand_geom)
    raise RuntimeError(f"Sampling failed after {max_samples} attempts")

def sample_move_away_from_target(
    obj_se3_pose: math_helpers.SE3Pose,
    robot_geom: Rectangle,
    collision_geoms: Collection[_Geom2D],
    rng: np.random.Generator,
    min_distance: float,
    max_distance: float,
    allowed_regions: Collection[scipy.spatial.Delaunay],  # pylint: disable=no-member
    max_samples: int = 1000
) -> Tuple[float, float, Rectangle]:
    """Sampler for navigating to a target object.

    Returns a distance and an angle in radians. Also returns the next
    robot geom for visualization and debugging convenience.
    """
    for _ in range(max_samples):
        # sampled angle is relative to the object frame
        selected_hull = rng.choice(allowed_regions)
        x, y = sample_point_in_hull(selected_hull, rng)
        distance = np.sqrt((x - obj_se3_pose.x) ** 2 + (y - obj_se3_pose.y) ** 2)
        angle = np.arctan2(y - obj_se3_pose.y, x - obj_se3_pose.x)
        if distance < min_distance or distance > max_distance:
            continue
        # Need to face towards the target. But has nothing to do with object yaw
        rot = angle + np.pi if angle < 0 else angle - np.pi
        cand_geom = Rectangle.from_center(x, y, robot_geom.width,
                                        robot_geom.height, rot)
        if valid_navigation_position(cand_geom, collision_geoms,
                                    allowed_regions):
            # All the angles are in -pi to pi range
            if angle > np.pi:
                angle -= 2 * np.pi
            if angle < -np.pi:
                angle += 2 * np.pi
            return (distance, angle, cand_geom)

    raise RuntimeError(f"Sampling failed after {max_samples} attempts")

def sample_hand_offset_from_target(
        obj_pose: math_helpers.SE3Pose,
        robot_pose: math_helpers.SE3Pose,
        rng: np.random.Generator,
        min_distance: float,
        max_distance: float,
        theta_tol=float,
        phi_tol=float,
        max_samples: int = 1000,
        ki_check: bool = False
) -> Tuple[float, float, float, float, float]:
    """Sampler for moving the hand relative to a target object.
    
    Returns the relative distance, the perturbed direction (x, y, z), and the YOZ angle.
    """
    for i in range(max_samples):
        # Sample a distance between min_distance and max_distance
        distance = rng.uniform(min_distance, max_distance)
        
        # Compute the base "look at" direction vector from the robot to the object
        obj_x_axis = obj_pose.rot.to_matrix()[:, 0]

        # get the theta and phi from direction vector
        theta = np.arcsin(obj_x_axis[2]) # OZ angle
        phi = np.arctan2(obj_x_axis[1], obj_x_axis[0]) # OX angle
        
        # Sample a small perturbation angle within the tolerance range
        theta_tol_rad = np.deg2rad(theta_tol)
        phi_tol_rad = np.deg2rad(phi_tol)
        perturbed_theta = rng.uniform(-theta_tol_rad, theta_tol_rad)
        perturbed_phi = rng.uniform(-phi_tol_rad, phi_tol_rad)
        
        # Compute perturbation in spherical coordinates around the base direction
        new_theta = theta + perturbed_theta
        new_phi = phi + perturbed_phi

        # All the angles are in -pi to pi range
        if new_theta > np.pi:
            new_theta -= 2 * np.pi
        if new_theta < -np.pi:
            new_theta += 2 * np.pi
        if new_phi > np.pi:
            new_phi -= 2 * np.pi
        if new_phi < -np.pi:
            new_phi += 2 * np.pi
        # Return the valid hand pose with perturbed direction
        if ki_check:
            logging.info(f"Sampling {i} times")
            # check if the new pose is valid
            new_x = obj_pose.x + distance * np.cos(new_theta) * np.cos(new_phi)
            new_y = obj_pose.y + distance * np.cos(new_theta) * np.sin(new_phi)
            new_z = obj_pose.z + distance * np.sin(new_theta)

            # Face towards the target. new x axis is the direction vector
            direction = np.array([obj_pose.x - new_x, obj_pose.y - new_y, obj_pose.z - new_z])
            new_x_axis = direction / np.linalg.norm(direction)

            tgt_z_axis = obj_pose.rot.to_matrix()[:, 2]
            # project the tgt_z_axis to the plane perpendicular to the new_x_axis
            tgt_z_axis = tgt_z_axis - np.dot(tgt_z_axis, new_x_axis) * new_x_axis
            new_z_axis = tgt_z_axis / np.linalg.norm(tgt_z_axis)

            new_y_axis = np.cross(new_z_axis, new_x_axis)
            new_rot = math_helpers.Quat.from_matrix(np.array([new_x_axis, new_y_axis, new_z_axis]).T)
            new_pose = math_helpers.SE3Pose(new_x, new_y, new_z, rot=new_rot)
            hand_pose_mat = new_pose.to_matrix()
            body_pose_mat = robot_pose.to_matrix()
            spot_arm_fk = SpotArmFK()
            suc, sol = spot_arm_fk.compute_ik(body_pose_mat, hand_pose_mat)
            if suc:
                return (distance, new_theta, new_phi)
        else:
            return (distance, new_theta, new_phi)
    raise RuntimeError(f"Sampling failed after {max_samples} attempts")

def get_robot_state(robot: Robot,
                    timeout_per_call: float = 20,
                    num_retries: int = 10) -> robot_state_pb2.RobotState:
    """Get the robot state."""
    robot_state_client = robot.ensure_client(
        RobotStateClient.default_service_name)
    for _ in range(num_retries):
        try:
            robot_state = robot_state_client.get_robot_state(
                timeout=timeout_per_call)
            return robot_state
        except (TimedOutError, ProxyConnectionError):
            logging.info("WARNING: get robot state failed once, retrying...")
    raise RuntimeError("get_robot_state() failed permanently.")


def get_robot_gripper_open_percentage(robot: Robot) -> float:
    """Get the current state of how open the gripper is."""
    robot_state = get_robot_state(robot)
    return float(robot_state.manipulator_state.gripper_open_percentage)

def spot_pose_to_geom2d(pose: math_helpers.SE3Pose) -> Rectangle:
    """Use known dimensions for spot robot to create a bounding box for the
    robot (top-down view).

    The origin of the rectangle is the back RIGHT leg of the spot.

    NOTE: the spot's x axis in the body frame points forward and the y axis
    points leftward. See the link below for an illustration of the frame.
    https://dev.bostondynamics.com/docs/concepts/geometry_and_frames
    """
    # We want to create a rectangle whose center is (pose.x, pose.y),
    # whose width (x direction) is front_to_back_length, whose height
    # (y direction) is side_length, and whose rotation is the pose yaw.
    front_to_back_length = 0.85  # meters, approximately
    side_length = 0.25
    yaw = pose.rot.to_yaw()
    return Rectangle.from_center(pose.x, pose.y, front_to_back_length,
                                 side_length, yaw)

def get_relative_se3_from_distance_and_angles(
    body_pose: math_helpers.SE3Pose,
    tgt_pose: math_helpers.SE3Pose,
    distance: float,
    theta: float,
    phi: float
) -> math_helpers.SE3Pose:
    """Given a current se3 pose and a target quaternion, return a relative se3
    pose for moving from the current to the target.

    Note that the quaternion is in the target frame.
    The returned is the target pose in the robot frame.
    """
    new_x = tgt_pose.x + distance * np.cos(theta) * np.cos(phi)
    new_y = tgt_pose.y + distance * np.cos(theta) * np.sin(phi)
    new_z = tgt_pose.z + distance * np.sin(theta)

    # Face towards the target. new x axis is the direction vector
    direction = np.array([tgt_pose.x - new_x, tgt_pose.y - new_y, tgt_pose.z - new_z])
    new_x_axis = direction / np.linalg.norm(direction)

    tgt_z_axis = tgt_pose.rot.to_matrix()[:, 2]
    # project the tgt_z_axis to the plane perpendicular to the new_x_axis
    tgt_z_axis = tgt_z_axis - np.dot(tgt_z_axis, new_x_axis) * new_x_axis
    new_z_axis = tgt_z_axis / np.linalg.norm(tgt_z_axis)

    new_y_axis = np.cross(new_z_axis, new_x_axis)
    new_rot = math_helpers.Quat.from_matrix(np.array([new_x_axis, new_y_axis, new_z_axis]).T)
    new_pose = math_helpers.SE3Pose(new_x, new_y, new_z, rot=new_rot)
    rel_pose = body_pose.inverse() * new_pose
    
    return rel_pose

def spot_stand(
    robot: Robot,
) -> None:
    """Stand up the robot."""
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), 'Robot power on failed.'
    robot.logger.info('Commanding robot to stand...')
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec=10)
    robot.logger.info('Robot standing.')

def spot_sit(
    robot: Robot,
) -> None:
    """Sit down the robot."""
    robot.logger.info('Commanding robot to sit...')
    robot.power_off(cut_immediately=False, timeout_sec=20)
    robot.logger.info('Robot sitting.')
    assert not robot.is_powered_on(), 'Robot power off failed.'
    robot.logger.info('Robot safely powered off.')

def add_noise_to_pose(
        original_pose: math_helpers.SE3Pose,
        rng: np.random.Generator,
        translation_noise: float = CFG.viewplan_trans_noise,
        rotation_noise: float = CFG.viewplan_angle_noise
    ) -> math_helpers.SE3Pose:
    """Add noise to the pose."""
    random_dir = rng.uniform(-1, 1, 3)
    random_dir /= np.linalg.norm(random_dir)
    random_translation = rng.uniform(0, translation_noise)
    new_x = original_pose.x + random_translation * random_dir[0]
    new_y = original_pose.y + random_translation * random_dir[1]
    new_z = original_pose.z + random_translation * random_dir[2]

    random_rot = rng.uniform(-rotation_noise, rotation_noise)
    # random rotation along z axis
    random_rot_mat = R.from_euler('xyz', [0, 0, random_rot]).as_matrix()
    new_rot = original_pose.rot.mult(math_helpers.Quat.from_matrix(random_rot_mat))
    return math_helpers.SE3Pose(new_x, new_y, new_z, rot=new_rot)

def sample_point_in_hull(delaunay, rng):
    """
    Sample a random point inside a convex hull using a Delaunay triangulation.
    
    Parameters:
        delaunay (scipy.spatial.Delaunay): Delaunay object for the convex hull.
        random_gen (np.random.Generator): NumPy random generator for sampling.
    
    Returns:
        np.ndarray: Randomly sampled point within the convex hull.
    """
    # Step 1: Randomly select a triangle
    simplices = delaunay.simplices
    simplex_index = rng.integers(len(simplices))  # Random triangle index
    simplex = simplices[simplex_index]

    # Step 2: Get the vertices of the selected triangle
    vertices = delaunay.points[simplex]

    # Step 3: Generate random barycentric coordinates
    barycentric = rng.random(3)
    barycentric /= barycentric.sum()  # Normalize to ensure they sum to 1

    # Step 4: Convert barycentric coordinates to Cartesian coordinates
    sampled_point = np.dot(barycentric, vertices)
    return sampled_point