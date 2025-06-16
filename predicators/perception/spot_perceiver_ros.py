"""A perceiver specific to spot envs."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import imageio.v2 as iio
import numpy as np
from bosdyn.client import math_helpers
from matplotlib import pyplot as plt

from predicators import utils
from predicators.envs import BaseEnv, get_or_create_env
from predicators.envs.view_plan import SpotViewPlanHardEnv, _SpotArmObservation
from predicators.envs.pick_place import SpotPickPlaceEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.settings import CFG
from predicators.spot_utils.utils import _stair_object_type, \
    _target_object_w_stair_type, _robot_hand_type, \
    get_allowed_map_regions, load_spot_metadata, object_to_top_down_geom
from predicators.spot_utils.utils_pick_place import _target_pickplace_type
from predicators.structs import Action, DefaultState, EnvironmentTask, \
    GoalDescription, GroundAtom, Object, Observation, Predicate, State, Task, \
    Video
from predicators.spot_utils.perception.object_detection import LangSAMModel, HumanLabeler


class SpotPerceiverROS(BasePerceiver):
    """A perceiver specific to spot envs."""

    def __init__(self) -> None:
        super().__init__()
        self._known_object_infos: Dict[Object, Dict] = {}
        self._objects_in_view: Set[Object] = set()
        self._objects_in_hand_view: Set[Object] = set()
        self._objects_in_any_view_except_back: Set[Object] = set()
        self._robot: Optional[Object] = None
        self._percept_predicates: Set[Predicate] = set()
        self._prev_action: Optional[Action] = None
        self._held_object: Optional[Object] = None
        self._gripper_open_percentage = 0.0
        self._robot_pos: math_helpers.SE3Pose = math_helpers.SE3Pose(
            0, 0, 0, math_helpers.Quat())
        self._lost_objects: Set[Object] = set()
        self._curr_env: Optional[BaseEnv] = None
        self._waiting_for_observation = True
        self._ordered_objects: List[Object] = []  # list of all known objects
        self._augment_info: Dict[str, tuple] = {}
        self._perceptor = None
        # if CFG.spot_run_dry:
        #     self._perceptor = None
        # else:
        #     if CFG.spot_perception_model == "LangSAM":
        #         self._perceptor = LangSAMModel()
        #     else:
        #         assert CFG.spot_perception_model == "Human"
        #         self._perceptor = HumanLabeler()
        # Keep track of objects that are contained (out of view) in another
        # object, like a bag or bucket. This is important not only for gremlins
        # but also for small changes in the container's perceived pose.
        self._container_to_contained_objects: Dict[Object, Set[Object]] = {}
        # Load static, hard-coded features of objects, like their shapes.
        meta = load_spot_metadata()
        self._static_object_features = meta.get("static-object-features", {})

    @classmethod
    def get_name(cls) -> str:
        return "spot_perceiver_ros"

    def reset(self, env_task: EnvironmentTask) -> Task:
        # Unless dry running, don't reset after the first time.
        if self._waiting_for_observation or CFG.spot_run_dry:
            self._waiting_for_observation = True
            self._curr_env = get_or_create_env(CFG.env)
            # assert isinstance(self._curr_env, ViewPlanHardEnv)
            self._known_object_infos = {}
            self._augment_info = {}
            self._objects_in_view = set()
            self._objects_in_hand_view = set()
            self._objects_in_any_view_except_back = set()
            self._robot = None
            self._percept_predicates = self._curr_env.predicates
            self._held_object = None
            self._gripper_open_percentage = 0.0
            self._robot_pos = math_helpers.SE3Pose(0, 0, 0,
                                                   math_helpers.Quat())
            self._robot_calib_id = -1
            self._robot_calibrated = False
            self._robot_hand_pos = math_helpers.SE3Pose(0, 0, 0,
                                        math_helpers.Quat())
            self._lost_objects = set()
            self._container_to_contained_objects = {}
        self._prev_action = None  # already processed at the end of the cycle
        if isinstance(self._curr_env, SpotViewPlanHardEnv):
            # This is a relatively fixed task
            init_state = self._create_state_viewplan()
        else:
            assert isinstance(self._curr_env, SpotPickPlaceEnv)
            init_state = self._create_state_pickplace()
        # This is a relatively fixed task
        goal = env_task.goal_description
        return Task(init_state, goal)

    def update_perceiver_with_action(self, action: Action) -> None:
        # NOTE: we need to keep track of the previous action
        # because the step function (where we need knowledge
        # of the previous action) occurs *after* the action
        # has already been taken.
        self._prev_action = action

    def step(self, observation: Observation) -> State:
        self._update_state_from_observation(observation)
        # Update the curr held item when applicable.
        assert self._curr_env is not None
        # Hack functions tracking prev_action, we don't use them for now.
        # if self._prev_action is not None:
        #     assert isinstance(self._prev_action.extra_info, (list, tuple))
        #     controller_name, objects, _, _ = self._prev_action.extra_info
        #     logging.info(
        #         f"[Perceiver] Previous action was {controller_name}{objects}.")
        #     # The robot is always the 0th argument of an
        #     # operator!
        #     if "pick" in controller_name.lower():
        #         if self._held_object is not None:
        #             assert CFG.spot_run_dry
        #         else:
        #             # We know that the object that we attempted to grasp was
        #             # the second argument to the controller.
        #             object_attempted_to_grasp = objects[1]
        #             # Remove from contained objects.
        #             for contained in self.\
        #                 _container_to_contained_objects.values():
        #                 contained.discard(object_attempted_to_grasp)
        #             # We only want to update the holding item id feature
        #             # if we successfully picked something.
        #             if self._gripper_open_percentage > \
        #                 HANDEMPTY_GRIPPER_THRESHOLD:
        #                 self._held_object = object_attempted_to_grasp
        #             else:
        #                 # We lost the object!
        #                 logging.info("[Perceiver] Object was lost!")
        #                 self._lost_objects.add(object_attempted_to_grasp)
        #     elif any(n in controller_name.lower() for n in
        #              ["place", "drop", "preparecontainerforsweeping", "drag"]):
        #         self._held_object = None
        #         # Check if the item we just placed is in view. It needs to
        #         # be in view to assess whether it was placed correctly.
        #         robot, obj = objects[:2]
        #         state = self._create_state()
        #         is_in_view = in_general_view_classifier(state, [robot, obj])
        #         if not is_in_view:
        #             # We lost the object!
        #             logging.info("[Perceiver] Object was lost!")
        #             self._lost_objects.add(obj)
        #     elif any(n in controller_name.lower()
        #              for n in ["sweepintocontainer", "sweeptwoobjects"]):
        #         robot = objects[0]
        #         state = self._create_state()
        #         if controller_name.lower() == "sweepintocontainer":
        #             objs = {objects[2]}
        #         else:
        #             assert controller_name.lower().startswith("sweeptwoobject")
        #             objs = {objects[2], objects[3]}
        #         for o in objs:
        #             is_in_view = in_general_view_classifier(state, [robot, o])
        #             if not is_in_view:
        #                 # We lost the object!
        #                 logging.info("[Perceiver] Object was lost!")
        #                 self._lost_objects.add(o)
        #     else:
        #         # Ensure the held object is reset if the hand is empty.
        #         prev_held_object = self._held_object
        #         if self._gripper_open_percentage <= HANDEMPTY_GRIPPER_THRESHOLD:
        #             self._held_object = None
        #             # This can only happen if the item was dropped during
        #             # something other than a place.
        #             if prev_held_object is not None:
        #                 # We lost the object!
        #                 logging.info("[Perceiver] An object was lost: "
        #                              f"{prev_held_object} was lost!")
        #                 self._lost_objects.add(prev_held_object)
        if isinstance(self._curr_env, SpotViewPlanHardEnv):
            # This is a relatively fixed task
            return self._create_state_viewplan()
        else:
            assert isinstance(self._curr_env, SpotPickPlaceEnv)
            return self._create_state_pickplace()

    def _update_state_from_observation(self, observation: Observation) -> None:
        assert isinstance(observation, _SpotArmObservation)
        self._waiting_for_observation = False
        self._robot = observation.robot
        self._known_object_infos.update(observation.objects_in_view)
        self._objects_in_view = set(observation.objects_in_view)
        self._objects_in_hand_view = observation.objects_in_hand_view
        self._objects_in_any_view_except_back = \
            observation.objects_in_any_view_except_back
        self._nonpercept_atoms = observation.nonpercept_atoms
        self._nonpercept_predicates = observation.nonpercept_predicates
        self._gripper_open_percentage = observation.gripper_open_percentage
        self._robot_pos = observation.robot_pos
        self._robot_hand_pos = observation.hand_pos
        self._robot_calibrated = observation.calibrated
        self._robot_calib_id = observation.calibration_id
        self._held_object = observation.object_in_hand

        # Run perception on images from observation.
        # rgbds = observation.images
        # if len(rgbds) > 0:
        #     logging.info(f"Running perception on {len(rgbds)} images.")
        #     pixel_infos = self._perceptor.segment(rgbds)
        #     if pixel_infos== None:
        #         self._augment_info = {}
        #     else:
        #         selected_image_name = list(pixel_infos.keys())[0]
        #         self._augment_info = {
        #             'rgbd': rgbds[selected_image_name],
        #             'pixels': pixel_infos[selected_image_name]
        #         }
        for obj in observation.objects_in_view:
            self._lost_objects.discard(obj)

    def _create_state_viewplan(self) -> State:
        if self._waiting_for_observation:
            return DefaultState
        # Build the continuous part of the state.
        assert self._robot is not None
        state_dict = {
            self._robot: {
                "gripper_open_percentage": self._gripper_open_percentage,
                "x": self._robot_pos.x,
                "y": self._robot_pos.y,
                "z": self._robot_pos.z,
                "qw": self._robot_pos.rot.w,
                "qx": self._robot_pos.rot.x,
                "qy": self._robot_pos.rot.y,
                "qz": self._robot_pos.rot.z,
                "ee_x": self._robot_hand_pos.x,
                "ee_y": self._robot_hand_pos.y,
                "ee_z": self._robot_hand_pos.z,
                "ee_qw": self._robot_hand_pos.rot.w,
                "ee_qx": self._robot_hand_pos.rot.x,
                "ee_qy": self._robot_hand_pos.rot.y,
                "ee_qz": self._robot_hand_pos.rot.z,
                "calibrated": self._robot_calibrated,
                "calibration_obj_id": self._robot_calib_id,
            },
        }
        # Add new objects to the list of known objects.
        known_objs = set(self._ordered_objects)
        for obj in sorted(set(self._known_object_infos) - known_objs):
            self._ordered_objects.append(None)
        for obj, info in self._known_object_infos.items():
            object_id = int(info["object_id"])
            pose = info["pose"]
            rgb = info["rgb"]
            self._ordered_objects[object_id] = obj
            state_dict[obj] = {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "qw": pose.rot.w,
                "qx": pose.rot.x,
                "qy": pose.rot.y,
                "qz": pose.rot.z,
                "shape": 1,
                "object_id": object_id,
                "r": rgb[0],
                "g": rgb[1],
                "b": rgb[2],
            }
            # Detect if the object is in (hand) view currently.
            if obj in self._objects_in_hand_view:
                in_hand_view_val = 1.0
            else:
                in_hand_view_val = 0.0
            state_dict[obj]["in_hand_view"] = in_hand_view_val
            if obj in self._objects_in_any_view_except_back:
                in_view_val = 1.0
            else:
                in_view_val = 0.0
            state_dict[obj]["in_view"] = in_view_val
            # Detect if we have lost the tool.
            if obj in self._lost_objects:
                lost_val = 1.0
            else:
                lost_val = 0.0
            state_dict[obj]["lost"] = lost_val
            # Add static object features.
            # static_feats = self._static_object_features.get(obj.name, {})
            # state_dict[obj].update(static_feats)
            # Add initial features for movable objects.
            if obj.is_instance(_stair_object_type):
                # Detect if the object is in (held) currently.
                if obj == self._held_object:
                    held_val = 1.0
                else:
                    held_val = 0.0
                state_dict[obj]["held"] = held_val
                # stair height
                state_dict[obj]["height"] = info["height"]
                # stair width and length
                state_dict[obj]["width"] = CFG.viewplan_stair_sz
                state_dict[obj]["length"] = CFG.viewplan_stair_sz
            elif obj.is_instance(_target_object_w_stair_type):
                # it can't be held
                # target width and length
                state_dict[obj]["width"] = 0.1
                state_dict[obj]["length"] = 0.1
                state_dict[obj]["height"] = 0.1
                state_dict[obj]["stair_id"] = info["stair_id"]
                state_dict[obj]["measured"] = info["measured"]
        # Construct a regular state before adding atoms.
        percept_state = utils.create_state_from_dict(state_dict)
        # Uncomment for debugging.
        # logging.info("Percept state:")
        # logging.info(percept_state.pretty_str())
        # logging.info("Percept atoms:")
        # atom_str = "\n".join(
        #     map(
        #         str,
        #         sorted(utils.abstract(percept_state,
        #                               self._percept_predicates))))
        # logging.info(atom_str)
        # logging.info("Simulator state:")
        # logging.info(simulator_state)

        # Now finish the state.
        # state = _AugmentPerceptionState(percept_state.data,
        #                 augmented_state=self._augment_info)
        return percept_state
    
    def _create_state_pickplace(self) -> State:
        if self._waiting_for_observation:
            return DefaultState
        # Build the continuous part of the state.
        assert self._robot is not None
        state_dict = {
            self._robot: {
                "gripper_open_percentage": self._gripper_open_percentage,
                "x": self._robot_pos.x,
                "y": self._robot_pos.y,
                "z": self._robot_pos.z,
                "qw": self._robot_pos.rot.w,
                "qx": self._robot_pos.rot.x,
                "qy": self._robot_pos.rot.y,
                "qz": self._robot_pos.rot.z,
                "ee_x": self._robot_hand_pos.x,
                "ee_y": self._robot_hand_pos.y,
                "ee_z": self._robot_hand_pos.z,
                "ee_qw": self._robot_hand_pos.rot.w,
                "ee_qx": self._robot_hand_pos.rot.x,
                "ee_qy": self._robot_hand_pos.rot.y,
                "ee_qz": self._robot_hand_pos.rot.z,
                "calibrated": self._robot_calibrated,
                "calibration_obj_id": self._robot_calib_id,
            },
        }
        # Add new objects to the list of known objects.
        known_objs = set(self._ordered_objects)
        for obj in sorted(set(self._known_object_infos) - known_objs):
            self._ordered_objects.append(None)
        for obj, info in self._known_object_infos.items():
            object_id = int(info["object_id"])
            pose = info["pose"]
            rgb = info["rgb"]
            self._ordered_objects[object_id] = obj
            state_dict[obj] = {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "qw": pose.rot.w,
                "qx": pose.rot.x,
                "qy": pose.rot.y,
                "qz": pose.rot.z,
                "shape": 1,
                "object_id": object_id,
                "r": rgb[0],
                "g": rgb[1],
                "b": rgb[2],
            }
            # Detect if the object is in (hand) view currently.
            if obj in self._objects_in_hand_view:
                in_hand_view_val = 1.0
            else:
                in_hand_view_val = 0.0
            state_dict[obj]["in_hand_view"] = in_hand_view_val
            if obj in self._objects_in_any_view_except_back:
                in_view_val = 1.0
            else:
                in_view_val = 0.0
            state_dict[obj]["in_view"] = in_view_val
            # Detect if we have lost the tool.
            if obj in self._lost_objects:
                lost_val = 1.0
            else:
                lost_val = 0.0
            state_dict[obj]["lost"] = lost_val
            # Add static object features.
            # static_feats = self._static_object_features.get(obj.name, {})
            # state_dict[obj].update(static_feats)
            # Add initial features for movable objects.
            if obj.is_instance(_stair_object_type):
                # Detect if the object is in (held) currently.
                if obj == self._held_object:
                    held_val = 1.0
                else:
                    held_val = 0.0
                state_dict[obj]["held"] = held_val
                # stair height
                state_dict[obj]["height"] = info["height"]
                # stair width and length
                state_dict[obj]["width"] = CFG.viewplan_stair_sz
                state_dict[obj]["length"] = CFG.viewplan_stair_sz
            elif obj.is_instance(_target_pickplace_type):
                if obj == self._held_object:
                    held_val = 1.0
                else:
                    held_val = 0.0
                state_dict[obj]["held"] = held_val
                state_dict[obj]["width"] = 0.1
                state_dict[obj]["length"] = 0.1
                state_dict[obj]["height"] = 0.1
                state_dict[obj]["stair_id"] = info["stair_id"]
                state_dict[obj]["achieved"] = info["achieved"]
        # Construct a regular state before adding atoms.
        percept_state = utils.create_state_from_dict(state_dict)
        # Uncomment for debugging.
        # logging.info("Percept state:")
        # logging.info(percept_state.pretty_str())
        # logging.info("Percept atoms:")
        # atom_str = "\n".join(
        #     map(
        #         str,
        #         sorted(utils.abstract(percept_state,
        #                               self._percept_predicates))))
        # logging.info(atom_str)
        # logging.info("Simulator state:")
        # logging.info(simulator_state)

        # Now finish the state.
        # state = _AugmentPerceptionState(percept_state.data,
        #                 augmented_state=self._augment_info)
        return percept_state

    def _create_goal(self, state: State) -> Set[GroundAtom]:
        # It is always measuring all the targets.
        pred_name_to_pred = {p.name: p for p in self._curr_env.predicates}
        goal_predicate = pred_name_to_pred["Measured"]
        goal = set()
        for obj in state.get_objects(_target_object_w_stair_type):
            goal.add(GroundAtom(goal_predicate, [obj]))
        return goal

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        pass
