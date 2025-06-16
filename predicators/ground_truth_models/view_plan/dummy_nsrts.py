"""Ground-truth NSRTs for the Spot View Planning environment."""

from typing import Dict, Sequence, Set, Tuple

import logging
import numpy as np

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotRearrangementEnv, \
    _get_sweeping_surface_for_container, get_detection_id_for_object
from predicators.ground_truth_models import DummyNSRTFactory
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    get_grasp_pixel, get_last_detected_objects
from predicators.spot_utils.perception.spot_cameras import \
    get_last_captured_images
from predicators.spot_utils.utils_trivial import get_allowed_map_regions, \
    get_collision_geoms_for_nav, load_spot_metadata, object_to_top_down_geom, \
    sample_move_offset_from_target, spot_pose_to_geom2d, sample_hand_offset_from_target, \
    get_viewable_geoms_for_nav
from predicators.structs import NSRT, Array, GroundAtom, NSRTSampler, Object, \
    ParameterizedOption, Predicate, State, Type, Variable, LiftedAtom


def _move_offset_sampler(state: State, robot_obj: Object,
                         obj_to_nav_to: Object, rng: np.random.Generator,
                         min_dist: float, max_dist: float, min_angle: float,
                         max_angle: float, obj_rel: bool=False) -> Array:
    """Called by all the different movement samplers."""
    obj_to_nav_to_pos = (state.get(obj_to_nav_to,
                        "x"), state.get(obj_to_nav_to, "y"))
    obj_to_nav_to_pose = utils.get_se3_pose_from_state(state, obj_to_nav_to)
    obj_to_nav_to_se2_pose = obj_to_nav_to_pose.get_closest_se2_transform()
    spot_pose = utils.get_se3_pose_from_state(state, robot_obj)
    robot_geom = spot_pose_to_geom2d(spot_pose)
    convex_hulls = get_allowed_map_regions()
    collision_geoms = get_collision_geoms_for_nav(state)
    try:
        abs_distance, abs_angle, _ = sample_move_offset_from_target(
            obj_to_nav_to_pos,
            robot_geom,
            collision_geoms,
            rng,
            min_distance=min_dist,
            max_distance=max_dist,
            allowed_regions=convex_hulls,
            obj_se2_pose=obj_to_nav_to_se2_pose,
            min_angle=min_angle,
            max_angle=max_angle,
            obj_rel=obj_rel,
        )
    # Rare sampling failures.
    except RuntimeError:  # pragma: no cover
        logging.info("WARNING: Failed to find good movement sample.")
        # Pick distance and angle at random.
        abs_distance = rng.uniform(min_dist, max_dist)
        abs_angle = rng.uniform(-np.pi, np.pi)
    # params are all normalized, they are unnormalized in the option
    distance = (abs_distance - min_dist) / (max_dist - min_dist)
    angle = abs_angle / np.pi # normalize angle
    return np.array([distance, angle])

def _move_offset_avoid_sampler(state: State, robot_obj: Object,
                            obj_to_nav_from: Object, rng: np.random.Generator,
                            min_dist: float, max_dist: float, min_angle: float,
                            max_angle: float) -> Array:
    """Called by all the move away from sampler."""
    obj_to_nav_from_pos = (state.get(obj_to_nav_from,
                        "x"), state.get(obj_to_nav_from, "y"))
    spot_pose = utils.get_se3_pose_from_state(state, robot_obj)
    robot_geom = spot_pose_to_geom2d(spot_pose)
    convex_hulls = get_allowed_map_regions()
    # we don't want to be in any of the viewable regions
    viewable_geoms = get_viewable_geoms_for_nav(state)

    try:
        abs_distance, abs_angle, _ = sample_move_offset_from_target(
            obj_to_nav_from_pos,
            robot_geom,
            viewable_geoms,
            rng,
            min_distance=min_dist,
            max_distance=max_dist,
            allowed_regions=convex_hulls,
            min_angle=min_angle,
            max_angle=max_angle,
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
            phi_tol=phi_tol
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


def _move_to_body_view_object_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del goal

    min_dist = 1.7
    max_dist = 1.95

    robot_obj = objs[0]
    obj_to_nav_to = objs[1]

    min_angle, max_angle = _get_approach_angle_bounds(obj_to_nav_to, state)

    return _move_offset_sampler(state, robot_obj, obj_to_nav_to, rng, min_dist,
                                max_dist, min_angle, max_angle)


def _move_to_hand_view_object_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del goal

    min_dist = CFG.cam2obj_distance_tol_trivial[0] + CFG.spot_arm_length_trivial
    max_dist = CFG.cam2obj_distance_tol_trivial[1] + CFG.spot_arm_length_trivial

    robot_obj = objs[0]
    obj_to_nav_to = objs[1]

    # relative to the object, face the obj x-y axis
    min_angle = -3 * np.pi / 2
    max_angle = -np.pi / 2

    return _move_offset_sampler(state, robot_obj, obj_to_nav_to, rng, min_dist,
                                max_dist, min_angle, max_angle, True)

def _move_away_from_object_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                        objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving away from).
    del goal

    min_dist = CFG.far_distance_trivial[0]
    max_dist = CFG.far_distance_trivial[1]

    robot_obj = objs[0]
    obj_to_get_away = objs[1]

    min_angle, max_angle = _get_approach_angle_bounds(obj_to_get_away, state)

    return _move_offset_avoid_sampler(state, robot_obj, obj_to_get_away, rng, min_dist,
                                max_dist, min_angle, max_angle)

def _hand_view_object_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, qx, qy, qz, qw (to the object you're trying to view).
    del goal

    min_dist = CFG.cam2obj_distance_tol_trivial[0]
    max_dist = CFG.cam2obj_distance_tol_trivial[1]

    theta_tol = CFG.cam2obj_angle_tol
    phi_tol = CFG.cam2obj_angle_tol

    obj_to_view = objs[1]
    return _hand_offset_sampler(state, objs[0], 
                                obj_to_view, rng, min_dist, 
                                max_dist, theta_tol, phi_tol)

def _move_to_reach_object_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del goal

    # NOTE: closer than move_to_view. Important for placing.
    min_dist = 0.1
    max_dist = 0.8

    robot_obj = objs[0]
    obj_to_nav_to = objs[1]

    min_angle, max_angle = _get_approach_angle_bounds(obj_to_nav_to, state)

    return _move_offset_sampler(state, robot_obj, obj_to_nav_to, rng, min_dist,
                                max_dist, min_angle, max_angle)


def _get_approach_angle_bounds(obj: Object,
                               state: State) -> Tuple[float, float]:
    """Helper for move samplers."""
    angle_bounds = load_spot_metadata().get("approach_angle_bounds", {})
    if obj.name in angle_bounds:
        return angle_bounds[obj.name]
    # Mega-hack for when the container is next to something with angle bounds,
    # i.e., it is ready to sweep.
    surface = _get_sweeping_surface_for_container(obj, state)
    if surface is not None and surface.name in angle_bounds:
        return angle_bounds[surface.name]
    # Default to all possible approach angles.
    return (-np.pi, np.pi)


def _pick_object_from_top_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    del state, goal  # not used
    target_obj = objs[1]
    # Special case: if we're running dry, the image won't be used.
    # Randomly sample a pixel.
    if CFG.spot_run_dry:
        # Load the object mask.
        if CFG.spot_use_perfect_samplers:
            obj_mask_filename = f"grasp_maps/{target_obj.name}-grasps.npy"
        else:
            obj_mask_filename = f"grasp_maps/{target_obj.name}-object.npy"
        obj_mask_path = utils.get_env_asset_path(obj_mask_filename)
        obj_mask = np.load(obj_mask_path)
        pixel_choices = np.where(obj_mask)
        num_choices = len(pixel_choices[0])
        choice_idx = rng.choice(num_choices)
        pixel_r = pixel_choices[0][choice_idx]
        pixel_c = pixel_choices[1][choice_idx]
        assert obj_mask[pixel_r, pixel_c]
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


def _place_object_on_top_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dz (to surface object's center)
    # in the WORLD FRAME.
    del goal
    surf_to_place_on = objs[2]
    surf_geom = object_to_top_down_geom(surf_to_place_on, state)
    if CFG.spot_use_perfect_samplers:
        if isinstance(surf_geom, utils.Rectangle):
            rand_x, rand_y = surf_geom.center
        else:
            assert isinstance(surf_geom, utils.Circle)
            rand_x, rand_y = surf_geom.x, surf_geom.y
    else:
        rand_x, rand_y = surf_geom.sample_random_point(rng, 0.13)
    dy = rand_y - state.get(surf_to_place_on, "y")
    if surf_to_place_on.name == "drafting_table":
        # For placing on the table, bias towards the top.
        # This makes a strong assumption about the world frame.
        # It may be okay to change these values, but one needs to be careful!
        assert abs(state.get(surf_to_place_on, "x") - 3.613) < 1e-3
        assert abs(state.get(surf_to_place_on, "y") + 0.908) < 1e-3
        dx = rng.uniform(0.1, 0.13)
    else:
        dx = rand_x - state.get(surf_to_place_on, "x")
    dz = 0.05
    # If we're placing the cup, we want to reduce the z
    # height for placing so the cup rests stably.
    if len(objs) == 3 and objs[1].name == "cup":
        dz = -0.05
    return np.array([dx, dy, dz])


def _drop_object_inside_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dz to the center of the top of the
    # container.
    del state, goal

    drop_height = 0.5
    if len(objs) == 4 and objs[2].name == "cup":
        drop_height = 0.05

    if CFG.spot_use_perfect_samplers:
        dx = 0.0
        dy = 0.0
    else:
        dx, dy = rng.uniform(-0.3, 0.3, size=2)

    return np.array([dx, dy, drop_height])


def _drag_to_unblock_object_sampler(state: State, goal: Set[GroundAtom],
                                    rng: np.random.Generator,
                                    objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dyaw to move while holding.
    del state, goal, objs, rng  # randomization coming soon
    return np.array([0.0, 0.0, np.pi])


def _drag_to_block_object_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dyaw to move while holding.
    del state, goal, objs, rng  # randomization coming soon
    return np.array([0.0, 0.0, -np.pi])


def _sweep_into_container_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Parameters are just one number, a velocity.
    del goal
    if CFG.spot_use_perfect_samplers:
        if CFG.spot_run_dry:
            if len(objs) == 6:  # SweepTwoObjectsIntoContainer
                _, _, target1, target2, _, container = objs
                targets = {target1, target2}
            else:
                assert len(objs) == 5  # SweepIntoContainer
                _, _, target, _, container = objs
                targets = {target}
            max_dist = 0.0
            cx, cy = state.get(container, "x"), state.get(container, "y")
            for target in targets:
                tx, ty = state.get(target, "x"), state.get(target, "y")
                dist = np.sum(np.square(np.subtract((cx, cy), (tx, ty))))
                max_dist = max(max_dist, dist)
            velocity = max_dist  # directly proportional
            return np.array([velocity])
        return np.array([1.0 / 0.58])
    if CFG.spot_run_dry:
        param = rng.uniform(0.1, 1.0)
    else:
        param = rng.uniform(0.1, 2.5)
    return np.array([param])


def _prepare_sweeping_sampler(state: State, goal: Set[GroundAtom],
                              rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:
    # Parameters are dx, dy, yaw w.r.t. the surface.
    del state, goal, rng, objs  # randomization coming soon
    param_dict = load_spot_metadata()["prepare_container_relative_xy"]
    return np.array([param_dict["dx"], param_dict["dy"], param_dict["angle"]])

class ViewPlanDummyNSRTFactory(DummyNSRTFactory):
    """Ground-truth NSRTs for the Spot View Planning environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"view_plan_trivial"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot_arm"]
        tgt_type = types["target"]

        # Predicates
        CalibrationTgt = predicates["CalibrationTgt"]
        Measured = predicates["Measured"]

        # Options
        MoveToHandView = options["MoveToHandViewObject"]
        MoveAwayFromObject = options["MoveAwayFromObject"]
        HandView = options["HandViewObject"]
        Calibrate = options["Calibrate"]
        Measure = options["Measure"]

        nsrts = set()

        # MoveToHandViewObject
        robot = Variable("?robot", robot_type)
        obj = Variable("?object", tgt_type)
        parameters = [robot, obj]
        option_vars = [robot, obj]
        option = MoveToHandView
        preconditions = {}
        add_effects = {}
        delete_effects = {}
        ignore_effects = set()

        movetosee_nsrt = NSRT("MoveToHandView", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           _move_to_hand_view_object_sampler)
        nsrts.add(movetosee_nsrt)

        # MoveAway
        robot = Variable("?robot", robot_type)
        obj = Variable("?object", tgt_type)
        parameters = [robot, obj]
        option_vars = [robot, obj]
        option = MoveAwayFromObject
        preconditions = {}
        add_effects = {}
        delete_effects = {}
        ignore_effects = set()

        moveaway_nsrt = NSRT("MoveAwayFromObject", parameters, preconditions, add_effects,
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
