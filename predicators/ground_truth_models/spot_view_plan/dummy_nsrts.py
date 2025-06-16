"""Ground-truth NSRTs for the Spot View Planning environment."""

from typing import Dict, Sequence, Set, List

import logging
import numpy as np
from bosdyn.client import math_helpers

from predicators import utils
from predicators.envs import get_or_create_env
from scipy.spatial.transform import Rotation as R
from predicators.ground_truth_models import DummyNSRTFactory
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    get_grasp_pixel, get_last_detected_objects
from predicators.spot_utils.kinematics.spot_arm import SpotArmFK
from predicators.spot_utils.perception.spot_cameras import \
    get_last_captured_images
from predicators.spot_utils.utils import get_allowed_map_regions, \
    get_collision_geoms_for_nav, sample_move_offset_with_ik, \
    sample_move_away_from_target, spot_pose_to_geom2d, sample_hand_offset_from_target, \
    get_close_geoms_for_nav
from predicators.structs import NSRT, Array, GroundAtom, NSRTSampler, Object, \
    ParameterizedOption, Predicate, State, Type, Variable, LiftedAtom


def _move_offset_ground_sampler(state: State, robot_obj: Object,
                         obj_to_nav_to: Object, rng: np.random.Generator,
                         dx: List[float], dy: List[float], d_angle: float,
                         objective: str, stair_height: float = 0.0) \
                         -> Array:
    """Called by all the different movement samplers. But always on the ground."""
    obj_to_nav_to_pose = utils.get_se3_pose_from_state(state, obj_to_nav_to)
    spot_pose = utils.get_se3_pose_from_state(state, robot_obj)
    body_height = spot_pose.z
    robot_geom = spot_pose_to_geom2d(spot_pose)
    convex_hulls = get_allowed_map_regions()
    collision_geoms = get_collision_geoms_for_nav(state)
    try:
        abs_dx, abs_dy, abs_dangle, _ = sample_move_offset_with_ik(
            obj_to_nav_to_pose,
            body_height,
            robot_geom,
            collision_geoms,
            rng,
            dx=dx,
            dy=dy,
            dtheta=d_angle,
            allowed_regions=convex_hulls,
            objective=objective,
            stair_height=stair_height,
            max_samples=10
        )

    except RuntimeError:  # pragma: no cover
        logging.info(f"WARNING: Failed to find good movement sample.")
        # On the last try, pick distance and angle at random.
        abs_dx = rng.uniform(min(dx), max(dx))
        abs_dy = rng.uniform(min(dy), max(dy))
        abs_dangle = rng.uniform(-np.pi, np.pi)
    # params are all normalized, they are unnormalized in the option
    if max(dx) == min(dx):
        assert abs_dx == 0
        normalized_dx = 0
    else:
        normalized_dx = (abs_dx - min(dx)) / (max(dx) - min(dx))
    if max(dy) == min(dy):
        assert abs_dy == 0
        normalized_dy = 0
    else:
        normalized_dy = (abs_dy - min(dy)) / (max(dy) - min(dy))
    normalized_dangle = abs_dangle / np.pi # normalize angle
    return np.array([normalized_dx, normalized_dy, normalized_dangle])

def _move_offset_avoid_sampler(state: State, robot_obj: Object,
                            obj_to_nav_from: Object, rng: np.random.Generator,
                            min_dist: float, max_dist: float) -> Array:
    """Called by all the move away from sampler."""
    obj_to_nav_from_pos = utils.get_se3_pose_from_state(state, obj_to_nav_from)
    spot_pose = utils.get_se3_pose_from_state(state, robot_obj)
    robot_geom = spot_pose_to_geom2d(spot_pose)
    convex_hulls = get_allowed_map_regions()
    # we don't want to be in any of the close regions
    close_geoms = get_close_geoms_for_nav(state)
    try:
        # note that max_dist is not used in sampler
        # as long as the distance is larger than min_dist
        abs_distance, abs_angle, _ = sample_move_away_from_target(
            obj_to_nav_from_pos,
            robot_geom,
            close_geoms,
            rng,
            min_distance=min_dist,
            max_distance=max_dist,
            allowed_regions=convex_hulls,
            max_samples=10
        )
    except RuntimeError:  # pragma: no cover
        logging.info("WARNING: Failed to find good movement sample.")
        # Pick distance and angle at random.
        abs_distance = rng.uniform(min_dist, max_dist)
        abs_angle = rng.uniform(-np.pi, np.pi)
    # params are all normalized, they are unnormalized in the option
    distance = (abs_distance - min_dist) / (max_dist - min_dist)
    angle = abs_angle / np.pi # normalize angle
    return np.array([distance, angle])

def _hand_offset_sampler(state: State, robot_obj: Object,
            obj_hand_sees: Object, rng: np.random.Generator,
            min_dist: float, max_dist: float, theta_tol: float, 
            phi_tol: float) -> Array:
    """Called by all the hand move samplers."""
    obj_pose = utils.get_se3_pose_from_state(state, obj_hand_sees)
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)
    try:
        abs_distance, abs_theta, abs_phi = \
        sample_hand_offset_from_target(
            obj_pose,
            robot_pose,
            rng,
            min_distance=min_dist,
            max_distance=max_dist,
            theta_tol=theta_tol,
            phi_tol=phi_tol,
            ki_check=True,
            max_samples=10
        )
    except RuntimeError:  # pragma: no cover
        logging.info("WARNING: Failed to find good hand pose sample.")
        # Pick distance and angle at random.
        abs_distance = rng.uniform(min_dist, max_dist)
        abs_theta = rng.uniform(-np.pi, np.pi)
        abs_phi = rng.uniform(-np.pi, np.pi)
    # params are all normalized, they are unnormalized in the option
    distance = (abs_distance - min_dist) / (max_dist - min_dist)
    theta = abs_theta / np.pi # normalize angle
    phi = abs_phi / np.pi # normalize angle
    return np.array([distance, theta, phi])

def _move_to_hand_view_object_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del goal
    robot_obj = objs[0]
    obj_to_nav_to = objs[1]

    tgt_yaw = utils.get_se3_pose_from_state(state, obj_to_nav_to).rot.to_yaw()
    dx_max = CFG.viewobj_distance[1] * np.cos(tgt_yaw)
    dx_min = CFG.viewobj_distance[0] * np.cos(tgt_yaw)
    dy_max = CFG.viewobj_distance[1] * np.sin(tgt_yaw)
    dy_min = CFG.viewobj_distance[0] * np.sin(tgt_yaw)

    objective = "handsee"
    dx = [dx_min, dx_max]
    dy = [dy_min, dy_max]
    # relative to the object, face the obj x-axis
    d_angle = CFG.move2hand_view_yaw_tol

    return _move_offset_ground_sampler(state, robot_obj, obj_to_nav_to, rng, dx,
                                dy, d_angle, objective)

def _move_away_from_object_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                        objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving away from).
    del goal

    min_dist = CFG.far_distance[0]
    max_dist = CFG.far_distance[1]

    robot_obj = objs[0]
    obj_to_get_away = objs[1]

    return _move_offset_avoid_sampler(state, robot_obj, obj_to_get_away, rng, min_dist,
                                max_dist)

def _hand_view_object_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, qx, qy, qz, qw (to the object you're trying to view).
    del goal

    min_dist = CFG.cam2obj_distance_tol_sampler[0]
    max_dist = CFG.cam2obj_distance_tol_sampler[1]

    theta_tol = CFG.cam2obj_angle_tol_hard_sampler
    phi_tol = CFG.cam2obj_angle_tol_hard_sampler

    obj_to_view = objs[1]
    return _hand_offset_sampler(state, objs[0], 
                                obj_to_view, rng, min_dist, 
                                max_dist, theta_tol, phi_tol)

def _move_to_reach_object_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del goal

    robot_obj = objs[0]
    obj_to_nav_to = objs[1]
    tgt_yaw = utils.get_se3_pose_from_state(state, obj_to_nav_to).rot.to_yaw()
    dx_max = CFG.reach_stair_distance[1] * np.cos(tgt_yaw)
    dx_min = CFG.reach_stair_distance[0] * np.cos(tgt_yaw)
    dy_max = CFG.reach_stair_distance[1] * np.sin(tgt_yaw)
    dy_min = CFG.reach_stair_distance[0] * np.sin(tgt_yaw)

    objective = "grasp"
    dx = [dx_min, dx_max]
    dy = [dy_min, dy_max]
    d_angle = CFG.reach_stair_yaw_tol

    return _move_offset_ground_sampler(state, robot_obj, obj_to_nav_to, rng, dx,
                                dy, d_angle, objective)

def _move_to_put_stair_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del goal

    # NOTE: at least half the stair width
    objective = "put_stair"

    robot_obj = objs[0]
    stair_in_hand = objs[1]
    stair_height = state.get(stair_in_hand, "height")
    obj_to_nav_to = objs[2]

    tgt_yaw = utils.get_se3_pose_from_state(state, obj_to_nav_to).rot.to_yaw()
    # Keep consistent with sampler
    dx_max = abs(CFG.put_stair_tgt_distance * np.cos(tgt_yaw))
    # dx_min = CFG.put_stair_tgt_distance[0] * np.cos(tgt_yaw)
    dy_max = abs(CFG.put_stair_tgt_distance * np.sin(tgt_yaw))
    # dy_min = CFG.put_stair_tgt_distance[0] * np.sin(tgt_yaw)
    dx = [-dx_max, dx_max]
    dy = [-dy_max, dy_max]
    d_angle = CFG.put_stair_tgt_yaw_tol

    return _move_offset_ground_sampler(state, robot_obj, obj_to_nav_to, rng, dx,
                                dy, d_angle, objective, stair_height)


def _pick_object_from_top_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    del state, goal  # not used
    target_obj = objs[1]
    # Special case: if we're running dry, the image won't be used.
    # Randomly sample a pixel.
    if CFG.spot_run_dry:
        # Load the object mask.
        # if CFG.spot_use_perfect_samplers:
        #     obj_mask_filename = f"grasp_maps/{target_obj.name}-grasps.npy"
        # else:
        #     obj_mask_filename = f"grasp_maps/{target_obj.name}-object.npy"
        # obj_mask_path = utils.get_env_asset_path(obj_mask_filename)
        # obj_mask = np.load(obj_mask_path)
        # pixel_choices = np.where(obj_mask)
        # num_choices = len(pixel_choices[0])
        # choice_idx = rng.choice(num_choices)
        # pixel_r = pixel_choices[0][choice_idx]
        # pixel_c = pixel_choices[1][choice_idx]
        pixel_r = rng.integers(0, 480)
        pixel_c = rng.integers(0, 640)
        # assert obj_mask[pixel_r, pixel_c]
        params_tuple = (pixel_r, pixel_c, 0.0, 0.0, 0.0, 0.0)
    else:
        # Select the coordinates of a pixel within the image so that
        # we grasp at that pixel!
        target_detection_id = get_detection_id_for_object(target_obj)
        rgbds = get_last_captured_images()
        _, artifacts = get_last_detected_objects()
        hand_camera = "hand_color_image"
        pixel, rot_quat = get_grasp_pixel(rgbds, artifacts,
                                          target_detection_id, hand_camera,
                                          rng)
        if rot_quat is None:
            rot_quat_tuple = (0.0, 0.0, 0.0, 0.0)
        else:
            rot_quat_tuple = (rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z)
        params_tuple = pixel + rot_quat_tuple

    return np.array(params_tuple)


# def _place_object_on_top_sampler(state: State, goal: Set[GroundAtom],
#                                  rng: np.random.Generator,
#                                  objs: Sequence[Object]) -> Array:
#     # Parameters are relative dx, dy, dz (to surface object's center)
#     # in the WORLD FRAME.
#     del goal
#     surf_to_place_on = objs[2]
#     surf_geom = object_to_top_down_geom(surf_to_place_on, state)
#     if CFG.spot_use_perfect_samplers:
#         if isinstance(surf_geom, utils.Rectangle):
#             rand_x, rand_y = surf_geom.center
#         else:
#             assert isinstance(surf_geom, utils.Circle)
#             rand_x, rand_y = surf_geom.x, surf_geom.y
#     else:
#         rand_x, rand_y = surf_geom.sample_random_point(rng, 0.13)
#     dy = rand_y - state.get(surf_to_place_on, "y")
#     if surf_to_place_on.name == "drafting_table":
#         # For placing on the table, bias towards the top.
#         # This makes a strong assumption about the world frame.
#         # It may be okay to change these values, but one needs to be careful!
#         assert abs(state.get(surf_to_place_on, "x") - 3.613) < 1e-3
#         assert abs(state.get(surf_to_place_on, "y") + 0.908) < 1e-3
#         dx = rng.uniform(0.1, 0.13)
#     else:
#         dx = rand_x - state.get(surf_to_place_on, "x")
#     dz = 0.05
#     # If we're placing the cup, we want to reduce the z
#     # height for placing so the cup rests stably.
#     if len(objs) == 3 and objs[1].name == "cup":
#         dz = -0.05
#     return np.array([dx, dy, dz])


class SpotViewPlanDummyNSRTFactory(DummyNSRTFactory):
    """Ground-truth NSRTs for the Spot View Planning environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"spot_view_plan_hard"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot_arm"]
        stair_type = types["stair"]
        tgt_type = types["target"]

        # Predicates
        DirectViewable = predicates["DirectViewable"]
        ViewClear = predicates["ViewClear"]
        AppliedTo = predicates["AppliedTo"]
        SurroundingClear = predicates["SurroundingClear"]
        CalibrationTgt = predicates["CalibrationTgt"]
        Measured = predicates["Measured"]

        # Options
        MoveToReach = options["MoveToReachObject"]
        Pick = options["PickObjectFromTop"]
        MoveToPlace = options["MoveToPlaceObject"]
        Place = options["PlaceObjectInFront"]
        MoveToOnStairsHandView = options["MoveToOnStairsHandViewObject"]
        MoveToHandView = options["MoveToHandViewObject"]
        MoveAwayFromObject = options["MoveAwayFromObject"]
        MoveAwayFromObjectStair = options["MoveAwayFromObjectStair"]
        HandView = options["HandViewObject"]
        Calibrate = options["Calibrate"]
        Measure = options["Measure"]

        nsrts = set()

        # MoveToReachObj
        robot = Variable("?robot", robot_type)
        stair = Variable("?stair", stair_type)

        parameters = [robot, stair]
        option_vars = [robot, stair]
        option = MoveToReach
        preconditions = {
            LiftedAtom(ViewClear, [robot]), # This is needed
            LiftedAtom(SurroundingClear, [robot]),
        }
        add_effects = {}
        delete_effects = {
            LiftedAtom(SurroundingClear, [robot]),
        }
        ignore_effects = set()
        movetoreach_nsrt = NSRT("MoveToPickObject", parameters, preconditions,
                               add_effects, delete_effects, ignore_effects,
                               option, option_vars, 
                               _move_to_reach_object_sampler)
        nsrts.add(movetoreach_nsrt)

        # PickObj
        robot = Variable("?robot", robot_type)
        stair = Variable("?stair", stair_type)

        parameters = [robot, stair]
        option_vars = [robot, stair]
        option = Pick
        preconditions = {}
        add_effects = {}
        delete_effects = {}
        ignore_effects = set()
        pick_nsrt = NSRT("PickObject", parameters, preconditions, add_effects,
                        delete_effects, ignore_effects, option, option_vars,
                        utils.null_sampler)
        nsrts.add(pick_nsrt)

        # MoveToPlaceObj
        robot = Variable("?robot", robot_type)
        stair = Variable("?stair", stair_type)
        obj = Variable("?object", tgt_type)

        parameters = [robot, stair, obj]
        option_vars = [robot, stair, obj]
        option = MoveToPlace

        preconditions = {
            LiftedAtom(AppliedTo, [stair, obj]),
            LiftedAtom(ViewClear, [robot]),
        }

        add_effects = {}
        delete_effects = set()
        ignore_effects = set()
        movetoplace_nsrt = NSRT("MoveToPlaceObject", parameters, preconditions,
                                add_effects, delete_effects, ignore_effects,
                                option, option_vars,
                                _move_to_put_stair_sampler)
        nsrts.add(movetoplace_nsrt)


        # PlaceObj
        robot = Variable("?robot", robot_type)
        stair = Variable("?stair", stair_type)
        obj = Variable("?object", tgt_type)

        parameters = [robot, stair, obj]
        option_vars = [robot, stair, obj]
        option = Place

        preconditions = {}
        add_effects = {}
        delete_effects = {}
        ignore_effects = set()
        place_nsrt = NSRT("PlaceObject", parameters, preconditions, add_effects,
                        delete_effects, ignore_effects, option, option_vars,
                        utils.null_sampler)
        nsrts.add(place_nsrt)


        # MoveToHandViewObject
        robot = Variable("?robot", robot_type)
        obj = Variable("?object", tgt_type)
        parameters = [robot, obj]
        option_vars = [robot, obj]
        option = MoveToHandView
        preconditions = {
            LiftedAtom(SurroundingClear, [robot]), # This is needed
            LiftedAtom(DirectViewable, [obj]),
            LiftedAtom(ViewClear, [robot]),
        }
        add_effects = {}
        delete_effects = {
            LiftedAtom(ViewClear, [robot]),
        }
        ignore_effects = set()

        movetosee_nsrt = NSRT("MoveToHandView", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           _move_to_hand_view_object_sampler)
        nsrts.add(movetosee_nsrt)

        # MoveToOnStairsHandViewObject
        robot = Variable("?robot", robot_type)
        stair = Variable("?stair", stair_type)
        obj = Variable("?object", tgt_type)
        parameters = [robot, stair, obj]
        option_vars = [robot, stair, obj]
        option = MoveToOnStairsHandView
        preconditions = {
            LiftedAtom(AppliedTo, [stair, obj]),
            LiftedAtom(ViewClear, [robot]),
        }
        add_effects = {}
        delete_effects = {
            LiftedAtom(ViewClear, [robot]),
        }
        ignore_effects = set()

        movetosee_nsrt = NSRT("MoveOnStairsHandView", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           utils.null_sampler)
        nsrts.add(movetosee_nsrt)

        # MoveAwayObj
        robot = Variable("?robot", robot_type)
        obj = Variable("?object", tgt_type)
        parameters = [robot, obj]
        option_vars = [robot, obj]
        option = MoveAwayFromObject
        preconditions = {
            LiftedAtom(Measured, [obj]),
        }
        add_effects = {
            LiftedAtom(ViewClear, [robot]),
        }
        delete_effects = {}
        ignore_effects = set()

        moveaway_nsrt = NSRT("MoveAwayFromObject", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           _move_away_from_object_sampler)
        nsrts.add(moveaway_nsrt)

        # MoveAwayObjStair
        robot = Variable("?robot", robot_type)
        stair = Variable("?stair", stair_type)
        obj = Variable("?object", tgt_type)
        parameters = [robot, stair, obj]
        option_vars = [robot, stair, obj]
        option = MoveAwayFromObjectStair
        preconditions = {
            LiftedAtom(Measured, [obj]),
        }
        add_effects = {
            LiftedAtom(SurroundingClear, [robot]),
            LiftedAtom(ViewClear, [robot]),
        }
        delete_effects = {}
        ignore_effects = set()

        moveaway_nsrt = NSRT("MoveAwayFromObjectStair", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           _move_away_from_object_sampler)
        nsrts.add(moveaway_nsrt)

        # HandSeeObject
        robot = Variable("?robot", robot_type)
        obj = Variable("?object", tgt_type)
        parameters = [robot, obj]
        option_vars = [robot, obj]
        option = HandView
        preconditions = {}
        add_effects = {}
        delete_effects = set()
        ignore_effects = set()

        armsee_nsrt = NSRT("HandView", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           _hand_view_object_sampler)
        nsrts.add(armsee_nsrt)

        # Calibrate
        robot = Variable("?robot", robot_type)
        obj = Variable("?object", tgt_type)
        parameters = [robot, obj]
        option_vars = [robot, obj]
        option = Calibrate
        preconditions = {
            LiftedAtom(CalibrationTgt, [robot, obj]),
        }
        add_effects = {}
        delete_effects = set()
        ignore_effects = set()
        calibrate_nsrt = NSRT("Calibrate", parameters, preconditions,
                              add_effects, delete_effects, ignore_effects,
                              option, option_vars, utils.null_sampler)
        nsrts.add(calibrate_nsrt)

        # Measure
        robot = Variable("?robot", robot_type)
        obj = Variable("?object", tgt_type)
        parameters = [robot, obj]
        option_vars = [robot, obj]
        option = Measure
        preconditions = {}
        add_effects = {
            LiftedAtom(Measured, [obj]),
        }
        delete_effects = set()
        ignore_effects = set()
        measure_nsrt = NSRT("Measure", parameters, preconditions, add_effects,
                            delete_effects, ignore_effects, option, option_vars,
                            utils.null_sampler)
        nsrts.add(measure_nsrt)

        return nsrts
