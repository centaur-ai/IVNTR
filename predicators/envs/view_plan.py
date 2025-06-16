"""A 3D View Planning Env for the real Spot Dog, partially inspired by SatellitesEnv.

The final goal is that the Spot use its arm camera measure some values
of several obejcts with specific view points.

To do this, the Spot will first find the calibration board, get calibrated,
then move to the objects, use arm to do view planning, and finally take the pictures.

An interesting part of this env is that, there exist stairs of different heights,
the Spot may need to move the stairs around and climb onto it to obtain the best view.
Obstacles are also placed in the environment to make the task more challenging.
Besides, the Spot could also move the targets into specific orientations so that
it can get the best view.

Note that this simulated environment will use the predicates and options/samplers 
(hand defined from real world) to generate demonstrations and tasks. There is only little 
kinematics and no physics of the real Spot. See Dry simulation function for details.
"""

import os
from natsort import natsorted
import logging
import time
import matplotlib.patches as mpatches
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Collection, Dict, Iterator, List, \
    Optional, Sequence, Set, Tuple

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from bosdyn.client import RetryableRpcError, create_standard_sdk, math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate, setup_logging
from gym.spaces import Box
from scipy.spatial import Delaunay

from predicators import utils
from predicators.utils import Rectangle
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.spot_utils.utils import _target_object_type, \
    _robot_hand_type, _stair_object_type, _target_object_w_stair_type, \
    DEFAULT_HAND2STAIR_TF, DEFAULT_STAIR2HAND_TF, \
    DEFAULT_DUMPED_TF,DEFAULT_HAND_LOOK_DOWN_POSE, DEFAULT_HAND_STOW_ANGLES, \
    DEFAULT_HAND_HOLDING_ANGLES, DEFAULT_STAIR2BODY_ONSTAIR_TF, \
    add_noise_to_pose, valid_navigation_position, \
    get_allowed_stair_regions, get_allowed_obj_regions
# helper functions, will need to implement using ROS2 interface
from predicators.spot_utils.perception.object_specific_grasp_selection import \
    brush_prompt, bucket_prompt, football_prompt, yogurt_prompt
from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_find_objects import \
    init_search_for_objects
from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import go_home, \
    navigate_to_absolute_pose, navigate_to_relative_pose_fixhand
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.perception.object_detection import \
    AprilTagObjectDetectionID, KnownStaticObjectDetectionID, \
    LanguageObjectDetectionID, ObjectDetectionID, detect_objects, \
    visualize_all_artifacts
# borrow predicates
from predicators.envs.spot_env import _NotBlocked, _HandEmpty, \
    _SpotArmObservation, _Viewable, _HandSees, get_robot, _hand_sees_hard_classifier, \
    _viewable_arm_classifier, _handempty_classifier, _reachable_classifier, \
    _SpotObservation
from predicators.structs import Action, EnvironmentTask, GoalDescription, \
    GroundAtom, LiftedAtom, Object, Observation, Predicate, State, \
    STRIPSOperator, Type
from predicators.spot_utils.kinematics.spot_arm import SpotArmFK
from predicators.spot_utils.utils import get_allowed_map_regions, load_spot_metadata, \
    sample_point_in_hull, get_robot_gripper_open_percentage, spot_stand

class ViewPlanTrivialEnv(BaseEnv):
    """A 3D continuous domain for Spot loosely inspired by the IPC domain of
    satellites. This is a trivial domain that only requires the Spot to move and handsee"""

    render_x_lb: ClassVar[float] = 0.0
    render_x_ub: ClassVar[float] = 8.0
    render_y_lb: ClassVar[float] = 0.0
    render_y_ub: ClassVar[float] = 8.0
    spot_radius: ClassVar[float] = 2.0 # used for initialization
    spot_height: ClassVar[float] = 0.94
    tgt_sz: ClassVar[float] = 0.1
    tgt_height_range: ClassVar[Tuple[float, float]] = (1.0, 1.2) # don't need a stair now
    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # assert CFG.spot_run_dry, "This environment is a dry run for the real Spot Dog."
        # assert CFG.spot_graph_nav_map == "debug", "This environment requires debug graph nav map."
        # Types
        ## "Spot" with an arm, "Stairs" of different heights, "Obstacle" of different shapes
        self._spot_type = _robot_hand_type
        # self._stairs_type = _movable_object_type
        self._tgt_obj_type = _target_object_type
        # Predicates
        self._HandSees = _HandSees
        self._NotBlocked = _NotBlocked
        self._Viewable = _Viewable
        self._CalibrationTgt = Predicate("CalibrationTgt", [self._spot_type, \
                                        self._tgt_obj_type], \
                                        self._CalibrationTgt_holds)
        self._Measured = Predicate("Measured", [self._tgt_obj_type],
                                       self._Measured_holds)
        self._Calibrated = Predicate("Calibrated", [self._spot_type],
                                        self._Calibrated_holds)
        # Quantified Not Viewable
        self._ViewClear = Predicate("ViewClear", [self._spot_type],
                                        self._ViewClear_holds)

    @classmethod
    def get_name(cls) -> str:
        return "view_plan_trivial"

    @property
    def action_space(self) -> Box:
        # The action space is effectively empty because only the extra info
        # part of actions are used.
        return Box(0, 1, (0, ))
    
    def step(self, action: Action) -> Observation:
        """Apply the action, update the state, and return an observation.

        Note that this action is a low-level action (i.e., action.arr
        is a member of self.action_space), NOT an option.

        By default, this function just calls self.simulate. However,
        environments that maintain a more complicated internal state,
        or that don't implement simulate(), may override this method.
        """
        # Note that this is a dry run env, so we use "State", instead of
        # "_SpotObservation"
        assert isinstance(self._current_observation, State)
        assert isinstance(action.extra_info, (list, tuple))
        action_name, action_objs, action_fn, action_fn_args = action.extra_info
        self._last_action = action
        self._current_observation = self.simulate(self._current_observation,
                                                  action)
        # Copy to prevent external changes to the environment's state.
        return self._current_observation.copy()
    
    def simulate(self, state: State, action: Action) -> State:
        assert CFG.spot_graph_nav_map == "debug", "This environment requires debug graph nav map."
        assert isinstance(action.extra_info, (list, tuple))
        action_name, action_objs, _, action_args = action.extra_info
        next_state = state.copy()
        # Note: This is a simulated env, not spot env, so we don't need to
        # care about "Obs" vs "State" here.
        if action_name == "MoveToHandViewObject" or \
            action_name == "MoveAwayFromObject":
            spot = action_objs[0]
            robot_rel_se2_pose = action_args[1]
            spot_pose = utils.get_se3_pose_from_state(next_state, spot)
            spot_z = spot_pose.z
            spot_se2_pose = spot_pose.get_closest_se2_transform()
            new_spot_se2_pose = spot_se2_pose * robot_rel_se2_pose
            # this assumes z is not changed
            new_spot_pose = new_spot_se2_pose.get_closest_se3_transform()
            new_spot_pose.z = spot_z
            next_state.set(spot, "x", new_spot_pose.x)
            next_state.set(spot, "y", new_spot_pose.y)
            next_state.set(spot, "z", new_spot_pose.z)
            next_state.set(spot, "qx", new_spot_pose.rot.x)
            next_state.set(spot, "qy", new_spot_pose.rot.y)
            next_state.set(spot, "qz", new_spot_pose.rot.z)
            next_state.set(spot, "qw", new_spot_pose.rot.w)
            # update the EE pose
            hand_pose = utils.get_se3_hand_pose_from_state(next_state, spot)
            # Assume the hand pose is relatively fixed to the spot pose
            new_hand_pose = new_spot_pose.mult(spot_pose.inverse().\
                                               mult(hand_pose))
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)

            return next_state
        elif action_name == "Calibrate":
            # first check if the spots hand sees the calibration target
            spot, obj = action_objs
            if not self._HandSees.holds(next_state, [spot, obj]):
                return next_state
            # then check if the tgt is the calibration tgt
            if not self._CalibrationTgt_holds(next_state, [spot, obj]):
                return next_state
            # finally, set the calibrated flag
            next_state.set(spot, "calibrated", 1)
            return next_state
        elif action_name == "Measure":
            # first check if the spots hand sees the target
            spot, obj = action_objs
            if not self._HandSees.holds(next_state, [spot, obj]):
                return next_state
            # then check if the spot is calibrated
            if not self._Calibrated_holds(next_state, [spot]):
                return next_state
            # finally, set the measured flag
            next_state.set(obj, "measured", 1)
            return next_state
        elif action_name == "HandViewObject":
            spot = action_objs[0]
            hand_rel_se3_pose = action_args[1]
            # update the EE pose
            hand_pose = utils.get_se3_hand_pose_from_state(next_state, spot)
            # Assume the hand pose is relatively fixed to the spot pose
            new_hand_pose = hand_pose * hand_rel_se3_pose
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
            return next_state

        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_obj_lst=CFG.viewplan_trivial_num_obj_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if CFG.in_domain_test:
            # use the same number of objects and stairs as training
            # no compositional generalization
            logging.info("Generating in-domain test tasks...")
            return self._get_tasks(num=CFG.num_test_tasks,
                                   num_obj_lst=CFG.viewplan_trivial_num_obj_train,
                                   rng=self._test_rng)
        else:
            logging.info("Generating out-of-domain test tasks...")
            return self._get_tasks(num=CFG.num_test_tasks,
                                num_obj_lst=CFG.viewplan_trivial_num_obj_test,
                                rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandSees, self._NotBlocked, self._ViewClear, self._Viewable, 
            self._Calibrated, self._CalibrationTgt, 
            self._Measured
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._Measured
        }

    @property
    def types(self) -> Set[Type]:
        return {self._spot_type, self._tgt_obj_type}

    def render_state_plt(self,
                        state: State,
                        task: EnvironmentTask,
                        action: Optional[Action] = None,
                        caption: Optional[str] = None,
                        save_path: Optional[str] = None):
        
        # Set the image resolution
        width, height = 1490, 1080  # Modify as needed

        # Create an offscreen renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height) # type: ignore
        
        # Set up scene
        scene = renderer.scene
        
        # World coordinates (x, y, z axes)
        world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])
        scene.add_geometry("world_frame", world_coordinate_frame, o3d.visualization.rendering.MaterialRecord()) # type: ignore
        
        # Load the RGB image
        plane = o3d.geometry.TriangleMesh.create_box(width=self.render_x_ub, height=self.render_y_ub, depth=0.01)

        # Optional: Assign a material to the box (if you want to color it, etc.)
        ground_material = o3d.visualization.rendering.MaterialRecord() # type: ignore
        ground_material.base_color = [0.55, 0.27, 0.07, 0.8]  # Set the box color (RGBA)

        # Add the plane to the scene with the material
        scene.add_geometry("plane", plane, ground_material)

        # Robot body frame (spot_robot)
        spot_robot = state.get_objects(self._spot_type)[0]
        
        spot_x = state.get(spot_robot, "x")
        spot_y = state.get(spot_robot, "y")
        spot_z = state.get(spot_robot, "z")
        spot_qx = state.get(spot_robot, "qx")
        spot_qy = state.get(spot_robot, "qy")
        spot_qz = state.get(spot_robot, "qz")
        spot_qw = state.get(spot_robot, "qw")
        
        # Convert quaternion to rotation matrix
        spot_rotation = R.from_quat([spot_qx, spot_qy, spot_qz, spot_qw]).as_matrix()
        spot_translation = np.array([spot_x, spot_y, spot_z])
        
        # Create robot body frame (as a coordinate frame)
        robot_body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8)
        robot_body_frame.translate(spot_translation)
        robot_body_frame.rotate(spot_rotation, center=spot_translation)
        scene.add_geometry("robot_body_frame", robot_body_frame, o3d.visualization.rendering.MaterialRecord()) # type: ignore

        # Define the box representing the robot (you can adjust size as needed)
        box_size = [0.85, 0.25, 0.1]  # Length, Width, Height of Spot Back
        robot_box = o3d.geometry.TriangleMesh.create_box(width=box_size[0], height=box_size[1], depth=box_size[2])

        # Translate the box so that the top plane is aligned with the XoY plane of the coordinate frame
        # This means the box's Z center will be shifted by half the box's height in the negative Z direction
        box_translation = np.array([spot_x, spot_y, 
                                    spot_z])

        # Apply a further translation to account for the box being created with its origin at a corner
        box_center_shift = np.array([box_size[0] / 2, box_size[1] / 2, box_size[2]])

        # Translate the box to the correct position
        robot_box.translate(box_translation - box_center_shift)
        robot_box.rotate(spot_rotation, center=box_translation)

        # Optional: Assign a material to the box (if you want to color it, etc.)
        box_material = o3d.visualization.rendering.MaterialRecord() # type: ignore
        box_material.base_color = [1.0, 1.0, 0.1, 1.0]  # Set the box color (RGBA)

        # Add the box to the scene
        scene.add_geometry("robot_box", robot_box, box_material)
        
        # Robot hand (camera) frame
        hand_x = state.get(spot_robot, "ee_x")
        hand_y = state.get(spot_robot, "ee_y")
        hand_z = state.get(spot_robot, "ee_z")
        hand_qx = state.get(spot_robot, "ee_qx")
        hand_qy = state.get(spot_robot, "ee_qy")
        hand_qz = state.get(spot_robot, "ee_qz")
        hand_qw = state.get(spot_robot, "ee_qw")
        robot_calibrated = state.get(spot_robot, "calibrated")
        
        hand_rotation = R.from_quat([hand_qx, hand_qy, hand_qz, hand_qw]).as_matrix()
        hand_translation = np.array([hand_x, hand_y, hand_z])
        # Create camera frame (as another coordinate frame)
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        camera_frame.translate(hand_translation)
        camera_frame.rotate(hand_rotation, center=hand_translation)
        scene.add_geometry("camera_frame", camera_frame, o3d.visualization.rendering.MaterialRecord()) # type: ignore

        # add the targets
        name2info = {}
        for tgt in state.get_objects(self._tgt_obj_type):
            tgt_x = state.get(tgt, "x")
            tgt_y = state.get(tgt, "y")
            tgt_z = state.get(tgt, "z")
            tgt_qx = state.get(tgt, "qx")
            tgt_qy = state.get(tgt, "qy")
            tgt_qz = state.get(tgt, "qz")
            tgt_qw = state.get(tgt, "qw")
            tgt_r = state.get(tgt, "r")
            tgt_g = state.get(tgt, "g")
            tgt_b = state.get(tgt, "b")
            tgt_rotation = R.from_quat([tgt_qx, tgt_qy, tgt_qz, tgt_qw]).as_matrix()
            tgt_translation = np.array([tgt_x, tgt_y, tgt_z])
            tgt_sz = state.get(tgt, "height")
            tgt_geom = o3d.geometry.TriangleMesh.create_box(width=tgt_sz, height=tgt_sz, depth=tgt_sz)
            tgt_shift = np.array([tgt_sz / 2, tgt_sz / 2, tgt_sz / 2])
            tgt_geom.translate(tgt_translation - tgt_shift)
            tgt_geom.rotate(tgt_rotation, center=tgt_translation)

            tgt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            tgt_frame.translate(tgt_translation)
            tgt_frame.rotate(tgt_rotation, center=tgt_translation)
            scene.add_geometry(f"{tgt.name}_frame", tgt_frame, o3d.visualization.rendering.MaterialRecord()) # type: ignore
            # add color to the target cube
            tgt_material = o3d.visualization.rendering.MaterialRecord() # type: ignore
            if tgt.name not in name2info:
                color = np.array([tgt_r, tgt_g, tgt_b]) / np.array([tgt_r, tgt_g, tgt_b]).max()
                name2info[tgt.name] = {
                    'color': list(color),
                    'measured': state.get(tgt, "measured")
                }
            tgt_material.base_color = list(color) + [1.0]  # Set the box color (RGBA)
            scene.add_geometry(tgt.name, tgt_geom, tgt_material)

        # List of different camera positions (you can add more views here)
        camera_positions = [
            [-self.render_x_ub/2, self.render_y_ub/2, 1], 
            [self.render_x_ub/2, self.render_y_ub/2, 6], 
            [-1, -1, 1],
            [self.render_x_ub/2, -self.render_y_ub/2, 1],  
        ]

        up_directions = [
            [0.1, 0, 0.9],   # Up direction for view 3
            [0, 1, 0],  # Up direction for view 1
            [0.1, 0.1, 0.9],
            [0, 0.1, 0.9],  # Up direction for view 2
        ]
        
        # Render multiple views and store images
        rendered_images = []  # Store rendered images here

        for i, camera_position in enumerate(camera_positions):
            # Set camera view
            camera = renderer.scene.camera
            camera.look_at(np.array([self.render_x_ub/2, self.render_y_ub/2, 0]), 
                           camera_position, up_directions[i])
            
            # Render the scene and capture the image
            image = renderer.render_to_image()
            
            # Convert Open3D image to a NumPy array (RGBA)
            image_np = np.asarray(image)
            
            # Store the NumPy image
            rendered_images.append(image_np)

        # Display the images in a single Matplotlib figure
        fig, axs = plt.subplots(2, 2, figsize=(90, 50))  # 2 row, 2 columns

        n = 0
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                # Display each image in a subplot
                ax.imshow(rendered_images[n])
                ax.axis('off')
                ax.set_title(f"View {n+1}")
                n += 1
    
        # Adjust the layout to provide more space at the bottom
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.98, bottom=0.25, left=0.01, right=0.99)

        # Create a new axis for the caption (position it at the bottom)
        caption_ax = fig.add_axes([0.3, 0.01, 0.9, 0.1])  # Position at the bottom within the figure bounds
        caption_ax.axis('off')  # Hide the axis

        # Create a list of colored squares (patches) and text for the caption
        patches = []
        labels = []
        # first add robot
        color_patch = mpatches.Rectangle((0, 0), 30, 30, facecolor=[1.0, 1.0, 0.2], 
                                         edgecolor="none")
        patches.append(color_patch)
        labels.append(f"Spot - C: {robot_calibrated}")
        for name, info in name2info.items():
            # Create a small colored square
            color_patch = mpatches.Rectangle((0, 0), 30, 30, facecolor=info['color'], edgecolor="none")
            patches.append(color_patch)
            labels.append(f"{name} - M: {info['measured']}")

        # Add the legend below the figure
        caption_ax.legend(patches, labels, loc='center', fontsize=50, frameon=False, ncol=len(name2info))

        # Optionally apply tight_layout as an additional adjustment
        # plt.tight_layout(rect=[0, 0, 0.1, 0.4])
        if save_path:
            fig.savefig(save_path, format='png', dpi=40)
        
        return fig
    
    def _get_tasks(self, num: int, num_obj_lst: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num):
            state_dict = {}
            color_id = 0
            num_obj = num_obj_lst[rng.choice(len(num_obj_lst))]
            # note that we use 2D circles to represent the 3D objects
            # just for collision checking
            collision_geoms: Set[utils._Geom2D] = set()
            # sample spot_body initial pose
            spot_robot = Object("spot", self._spot_type)
            spot_x = rng.uniform() * (self.render_x_ub - self.spot_radius)
            spot_y = rng.uniform() * (self.render_y_ub - self.spot_radius)
            spot_z = self.spot_height
            # random orientation on xy plane
            theta = rng.uniform(0.0, 2 * np.pi)  # Random yaw angle in radians
            spot_rot = math_helpers.Quat.from_yaw(theta)
            # world 2 body
            body_pose = math_helpers.SE3Pose(x=spot_x, y=spot_y, z=spot_z,
                    rot=spot_rot)
            spot_qx = body_pose.rot.x
            spot_qy = body_pose.rot.y
            spot_qz = body_pose.rot.z
            spot_qw = body_pose.rot.w
            # sample arm initial pose, body 2 hand
            hand_pose = body_pose.mult(DEFAULT_HAND_LOOK_DOWN_POSE)
            hand_x = hand_pose.x
            hand_y = hand_pose.y
            hand_z = hand_pose.z
            hand_qx = hand_pose.rot.x
            hand_qy = hand_pose.rot.y
            hand_qz = hand_pose.rot.z
            hand_qw = hand_pose.rot.w
            calibration_tgt = rng.choice(num_obj)
            state_dict[spot_robot] = {
                "gripper_open_percentage": 0.0, # initially always closed
                "x": spot_x, "y": spot_y, "z": spot_z,
                "qx": spot_qx, "qy": spot_qy, "qz": spot_qz, "qw": spot_qw,
                "ee_x": hand_x, "ee_y": hand_y, "ee_z": hand_z,
                "ee_qx": hand_qx, "ee_qy": hand_qy, "ee_qz": hand_qz, "ee_qw": hand_qw,
                "calibration_obj_id": calibration_tgt,
                "calibrated": 0,
            }
            # we use a large circle to represent the spot robot so that
            # it initially is viewclear
            geom = utils.Circle(spot_x, spot_y, 2.0)
            collision_geoms.add(geom)
            # sample tgt objects
            tgts = [Object(f"tgt{i}", self._tgt_obj_type) for i in range(num_obj)]
            # Sample initial positions for satellites, making sure to keep
            # them far enough apart from one another.
            for i, tgt in enumerate(tgts):
                # Assuming that the dimensions are forgiving enough that
                # infinite loops are impossible.
                while True:
                    x = rng.uniform() * (self.render_x_ub - self.tgt_sz)
                    y = rng.uniform() * (self.render_y_ub - self.tgt_sz)
                    # sample a random pose in xoy plane
                    theta = rng.uniform(0.0, 2 * np.pi)  # Random yaw angle in radians
                    qx = 0.0
                    qy = 0.0
                    qz = np.sin(theta / 2.0)
                    qw = np.cos(theta / 2.0)
                    rot = math_helpers.Quat(qw, qx, qy, qz) # type: ignore
                    angle = rot.to_yaw()
                    geom = utils.Rectangle.from_center(x, y, self.tgt_sz, self.tgt_sz,
                                           angle)
                    # Keep only if no intersections with existing objects.
                    if not any(geom.intersects(g) for g in collision_geoms):
                        break
                collision_geoms.add(geom)
                z = rng.uniform() * (self.tgt_height_range[1] - self.tgt_height_range[0]) \
                            + self.tgt_height_range[0]
                r, g, b = rng.uniform(size=3)
                state_dict[tgt] = {
                    "x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw,
                    "shape": 1,
                    "height": self.tgt_sz,
                    "width": self.tgt_sz,
                    "length": self.tgt_sz,
                    "object_id": i,
                    "measured": 0,
                    "r": r, "g": g, "b": b
                }
            init_state = utils.create_state_from_dict(state_dict)
            goal = set()
            for tgt in tgts:
                goal.add(GroundAtom(self._Measured, [tgt]))
            task = EnvironmentTask(init_state, goal)
            tasks.append(task)
        return tasks

    def _Measured_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj = objects[0]
        return bool(state.get(obj, "measured"))
    
    def _CalibrationTgt_holds(self, state: State, objects: Sequence[Object]) -> bool:
        spot = objects[0]
        tgt = objects[1]
        tgt_id = state.get(tgt, "object_id")
        spot_cal_id = state.get(spot, "calibration_obj_id")
        return spot_cal_id == tgt_id
    
    def _Calibrated_holds(self, state: State, objects: Sequence[Object]) -> bool:
        spot = objects[0]
        return bool(state.get(spot, "calibrated"))
    
    def _ViewClear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # This is required otherwise Viewable will be remembered
        spot = objects[0]
        for tgt in state.get_objects(self._tgt_obj_type):
            if self._Viewable.holds(state, [spot, tgt]):
                return False
        return True
    
class ViewPlanHardEnv(ViewPlanTrivialEnv):
    """A 3D continuous domain for Spot loosely inspired by the IPC domain of
    satellites."""
    render_x_lb: ClassVar[float] = -2.4
    render_x_ub: ClassVar[float] = 3.8
    render_y_lb: ClassVar[float] = -3.8
    render_y_ub: ClassVar[float] = 5.2
    spot_radius: ClassVar[float] = 2.0 # used for initialization
    ground_z: ClassVar[float] = CFG.viewplan_ground_z # measured
    spot_height: ClassVar[float] = CFG.viewplan_spot_height # measured
    spot_length: ClassVar[float] = CFG.spot_body_length
    tgt_sz: ClassVar[float] = 0.1
    tgt_height_range: ClassVar[Tuple[float, float]] = (1.0, 1.2) # for tgt that doesn't need stairs
    high_tgt_height_range: ClassVar[Tuple[float, float]] = (1.5, 1.75)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # assert CFG.spot_run_dry, "This environment is a dry run for the real Spot Dog."
        assert CFG.spot_graph_nav_map == "sqh_final", "This environment requires sqh_final graph nav map."
        self.spot_kinematics = SpotArmFK()
        self.visualize_joints = CFG.viewplan_visualize_joints
        self.stair_height_range = CFG.viewplan_stair_height_range
        self.stair_sz = CFG.viewplan_stair_sz
        # Types
        ## "Spot" with an arm, "Stairs" of different heights, "Obstacle" of different shapes
        self._spot_type = _robot_hand_type
        self._stairs_type = _stair_object_type
        self._tgt_obj_type = _target_object_w_stair_type
        # Predicates
        self._HandSees = Predicate("HandSees", [self._spot_type, self._tgt_obj_type], 
                     _hand_sees_hard_classifier)
        self._Viewable_Arm = Predicate("ViewableArm", [self._spot_type, self._tgt_obj_type],
                       _viewable_arm_classifier) # considers arm workspace
        # If the target is directly viewable by the spot w/o a stair
        self._DirectViewable = Predicate("DirectViewable", [self._tgt_obj_type],
                                         self._DirectViewable_holds)
        # If the stair can be used to stand on to handsee the target
        self._AppliedTo = Predicate("AppliedTo", [self._stairs_type, self._tgt_obj_type],
                                    self._AppliedTo_holds)
        # If the spot is holding the stair
        self._Holding = Predicate("Holding", [self._spot_type, self._stairs_type],
                                    self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._spot_type], _handempty_classifier)
        self._Reachable = Predicate("Reachable", [self._spot_type, self._stairs_type],
                       _reachable_classifier)
        # If the spot can't reach any stairs
        self._SurroundingClear = Predicate("SurroundingClear", [self._spot_type],
                                             self._SurroundingClear_holds)
        # If the stair is near the target
        self._Near = Predicate("Near", [self._stairs_type, self._tgt_obj_type],
                                    self._Near_holds)
        # If the spot is ready to put the stair down in front of the target
        self._Close = Predicate("Close", [self._spot_type, self._tgt_obj_type],
                                    self._Close_holds)
        # If the spot is on the ground, such that it can move
        self._OnGround = Predicate("OnGround", [self._spot_type],
                                    self._OnGround_holds)
        self._OnStair = Predicate("OnStair", [self._spot_type, self._stairs_type],
                                    self._OnStair_holds)
        self._CalibrationTgt = Predicate("CalibrationTgt", [self._spot_type, \
                                        self._tgt_obj_type], \
                                        self._CalibrationTgt_holds)
        self._Measured = Predicate("Measured", [self._tgt_obj_type],
                                       self._Measured_holds)
        self._Calibrated = Predicate("Calibrated", [self._spot_type],
                                        self._Calibrated_holds)
        # Quantified Not Viewable
        self._ViewClear = Predicate("ViewClear", [self._spot_type],
                                        self._ViewClear_holds)
        
        # pre-define all the distinguishable colors
        self.all_colors = [
            [0.0, 0.0, 1.0],  # blue
            [0.0, 1.0, 0.0],  # green
            [1.0, 0.0, 0.0],  # red
            [1.0, 1.0, 0.0],  # yellow
            [1.0, 0.0, 1.0],  # magenta
            [0.0, 1.0, 1.0],  # cyan
            [0.5, 0.5, 0.5],  # gray
            [1.0, 0.5, 0.0],  # orange
            [0.5, 0.0, 1.0],  # purple
            [0.0, 0.5, 1.0],  # light blue
            [0.0, 1.0, 0.5],  # light green
            [1.0, 0.0, 0.5],  # pink
            [0.5, 1.0, 0.0],  # lime
            [0.5, 0.0, 0.5],  # dark purple
            [0.0, 0.5, 0.5],  # dark cyan
            [0.5, 0.5, 0.0],  # olive
            [0.5, 0.5, 1.0],  # light purple
            [0.5, 1.0, 0.5],  # light lime
            [1.0, 0.5, 0.5],  # light pink
            [0.5, 0.5, 0.5],  # light gray
            [0.0, 0.0, 0.0],  # black
            [1.0, 1.0, 1.0],  # white
        ]
        
    @classmethod
    def get_name(cls) -> str:
        return "view_plan_hard"
    
    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandSees, self._ViewClear, self._Viewable_Arm, 
            self._Calibrated, self._CalibrationTgt, self._AppliedTo, self._HandEmpty,
            self._Holding, self._Near, self._Close, self._DirectViewable,
            self._OnGround, self._OnStair, self._Reachable, self._SurroundingClear,
            self._Measured
        }
    
    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._Measured
        }

    @property
    def types(self) -> Set[Type]:
        return {self._spot_type, self._tgt_obj_type, self._stairs_type}
    
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_obj_lst=CFG.viewplan_num_obj_train,
                               num_stairs_lst=CFG.viewplan_num_stairs_train,
                               rng=self._train_rng,
                               test=False)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if CFG.in_domain_test:
            # use the same number of objects and stairs as training
            # no compositional generalization
            logging.info("Generating in-domain test tasks...")
            return self._get_tasks(num=CFG.num_test_tasks,
                                   num_obj_lst=CFG.viewplan_num_obj_train,
                                   num_stairs_lst=CFG.viewplan_num_stairs_train,
                                   rng=self._test_rng,
                                   test=False)
        else:
            logging.info("Generating out-of-domain test tasks...")
            return self._get_tasks(num=CFG.num_test_tasks,
                                num_obj_lst=CFG.viewplan_num_obj_test,
                                num_stairs_lst=CFG.viewplan_num_stairs_test,
                                rng=self._test_rng,
                                test=True)

    def _get_tasks(self, num, num_obj_lst, num_stairs_lst, rng, test) \
        -> List[EnvironmentTask]:
        tasks = []
        task_id = 0
        convex_hulls = get_allowed_map_regions()
        while task_id < num:
            start_t = time.time()
            skip = False
            logging.info(f"Generating task {task_id}...")
            color_id = 0
            state_dict = {}
            num_stairs = num_stairs_lst[rng.choice(len(num_stairs_lst))]
            while True:
                num_obj = num_obj_lst[rng.choice(len(num_obj_lst))]
                if (num_obj >= num_stairs):
                    # make sure more targets than stairs
                    break
            # First, sample the spot robot
            collision_geoms: Set[utils._Geom2D] = set()
            # sample spot_body initial pose
            spot_robot = Object("spot", self._spot_type)
            while True:
                spot_x = rng.uniform() * (self.render_x_ub - self.render_x_lb) \
                            + self.render_x_lb
                spot_y = rng.uniform() * (self.render_y_ub - self.render_y_lb) \
                            + self.render_y_lb
                theta = rng.uniform(0.0, 2 * np.pi)  # Random yaw angle in radians
                cand_geom = Rectangle.from_center(spot_x, spot_y, 0.85,
                                        0.25, theta)
                if valid_navigation_position(
                    cand_geom, collision_geoms, convex_hulls):
                    break
            spot_z = self.spot_height + self.ground_z
            # random orientation on xy plane
            spot_rot = math_helpers.Quat.from_yaw(theta)
            body_pose = math_helpers.SE3Pose(x=spot_x, y=spot_y, z=spot_z,
                    rot=spot_rot)
            body_pose = add_noise_to_pose(body_pose, rng)
            spot_qx = body_pose.rot.x
            spot_qy = body_pose.rot.y
            spot_qz = body_pose.rot.z
            spot_qw = body_pose.rot.w
            # sample arm initial pose, body 2 hand
            body_pose_mat = body_pose.to_matrix()
            hand_pose_mat = self.spot_kinematics.compute_fk(body_pose_mat, \
                                            DEFAULT_HAND_STOW_ANGLES)[-1]
            hand_pose = math_helpers.SE3Pose.from_matrix(hand_pose_mat)
            hand_pose = add_noise_to_pose(hand_pose, rng)
            hand_x = hand_pose.x
            hand_y = hand_pose.y
            hand_z = hand_pose.z
            hand_qx = hand_pose.rot.x
            hand_qy = hand_pose.rot.y
            hand_qz = hand_pose.rot.z
            hand_qw = hand_pose.rot.w
            calibration_tgt = rng.choice(num_obj)
            state_dict[spot_robot] = {
                "gripper_open_percentage": 1.0, # Impotantly, initially always Opened
                "x": spot_x, "y": spot_y, "z": spot_z,
                "qx": spot_qx, "qy": spot_qy, "qz": spot_qz, "qw": spot_qw,
                "ee_x": hand_x, "ee_y": hand_y, "ee_z": hand_z,
                "ee_qx": hand_qx, "ee_qy": hand_qy, "ee_qz": hand_qz, "ee_qw": hand_qw,
                "calibration_obj_id": calibration_tgt,
                "calibrated": 0,
            }
            # we use a large circle to represent the spot robot so that
            # it initially is viewclear
            geom = utils.Circle(spot_x, spot_y, 1.5)
            collision_geoms.add(geom)
            # sample tgt objects
            tgts = [Object(f"tgt{i}", self._tgt_obj_type) for i in range(num_obj)]
            stairs = [Object(f"stair{i}", self._stairs_type) for i in range(num_stairs)]
            # Sample initial poses for targets, making sure to keep
            # them far enough apart from one another.
            initialized_tgt = []
            for i, stair in enumerate(stairs):
                # Step 1: sample stairs and their heights
                while True:
                    selected_hull = rng.choice(convex_hulls)
                    x, y = sample_point_in_hull(selected_hull, rng)
                    stair_height = rng.uniform(self.stair_height_range[0], self.stair_height_range[1])
                    geom = utils.Circle(x, y, 1.5)
                    if not any(geom.intersects(g) for g in collision_geoms):
                        suc, qx, qy, qz, qw = self.sample_stair_using_arm(rng, convex_hulls, x, y, stair_height)
                        if suc:
                            break
                # stair is not added to collision geoms, since the targets are floating higher
                collision_geoms.add(geom)
                r, g, b = self.all_colors[color_id]
                color_id += 1
                state_dict[stair] = {
                    "x": x, "y": y, "z": stair_height + self.ground_z,
                    "qx": qx, "qy": qy, "qz": qz, "qw": qw,
                    "shape": 1,
                    "height": stair_height,
                    "width": self.stair_sz,
                    "length": self.stair_sz,
                    "object_id": num_obj + i, # this is stair id, after tgt ids
                    "held": 0.0, 
                    "lost": 0.0, # dry sim always false
                    "in_hand_view": 0.0, # dry sim always false
                    "in_view": 1.0, # dry sim always true
                    "r": r, "g": g, "b": b
                }
                # Step 2: sample a target that needs the stair to be measured
                consider_tgts = []
                for tgt_id, tgt in enumerate(tgts):
                    if tgt in initialized_tgt:
                        # one stair for one target
                        continue
                    consider_tgts.append(tgt)
                stair_tgt = rng.choice(consider_tgts)
                # first sample its xy location
                while True:
                    if time.time() - start_t > 15:
                        logging.warning(f"Task {task_id} took too long to generate.")
                        skip = True
                        break
                    # use stair size, as we will put the stairs at a correct location
                    selected_hull = rng.choice(convex_hulls)
                    x, y = sample_point_in_hull(selected_hull, rng)
                    # targets are far enough from each other
                    geom = utils.Circle(x, y, 1.5)
                    # then sample its z location and orientation using stair height
                    # Note that tgt_z already adds ground_z
                    if not any(geom.intersects(g) for g in collision_geoms):
                        suc, tgt_z, qw, qx, qy, qz = \
                            self.sample_tgt_using_arm(rng, convex_hulls, tgt_x=x, tgt_y=y, \
                                                      stair_height=stair_height, \
                                                        off_stair=False)
                        if suc:
                            break
                if skip:
                    break
                collision_geoms.add(geom)
                r, g, b = self.all_colors[color_id]
                color_id += 1
                tgt_id = int(stair_tgt.name[3])
                state_dict[stair_tgt] = {
                    "x": x, "y": y, "z": tgt_z,
                    "qx": qx, "qy": qy, "qz": qz, "qw": qw,
                    "shape": 1,
                    "height": self.tgt_sz,
                    "width": self.tgt_sz,
                    "length": self.tgt_sz,
                    "object_id": tgt_id,
                    "stair_id": num_obj + i,
                    "measured": 0,
                    "lost": 0.0, # dry sim always false
                    "in_hand_view": 0.0, # dry sim always false
                    "in_view": 1.0, # dry sim always true
                    "r": r, "g": g, "b": b
                }
                initialized_tgt.append(stair_tgt)
            
            for tgt in tgts:
                if skip:
                    break
                if tgt in initialized_tgt:
                    continue
                # sample a target that doesn't need the stair
                while True:
                    if time.time() - start_t > 15:
                        logging.warning(f"Task {task_id} took too long to generate.")
                        skip = True
                        break
                    selected_hull = rng.choice(convex_hulls)
                    x, y = sample_point_in_hull(selected_hull, rng)
                    # targets are far enough from each other
                    geom = utils.Circle(x, y, 1.5)
                    if not any(geom.intersects(g) for g in collision_geoms):
                        suc, tgt_z, qw, qx, qy, qz = \
                            self.sample_tgt_using_arm(rng, convex_hulls, tgt_x=x, tgt_y=y, \
                                                      stair_height=0, 
                                                        off_stair=True)
                        if suc:
                            break
                if skip:
                    break
                collision_geoms.add(geom)
                r, g, b = self.all_colors[color_id]
                color_id += 1
                tgt_id = int(tgt.name[3])
                state_dict[tgt] = {
                    "x": x, "y": y, "z": tgt_z, "qx": qx, "qy": qy, "qz": qz, "qw": qw,
                    "shape": 1,
                    "height": self.tgt_sz,
                    "width": self.tgt_sz,
                    "length": self.tgt_sz,
                    "object_id": tgt_id,
                    "measured": 0,
                    "stair_id": -1,
                    "lost": 0.0, # dry sim always false
                    "in_hand_view": 0.0, # dry sim always false
                    "in_view": 1.0, # dry sim always true
                    "r": r, "g": g, "b": b
                }

            if not skip:
                init_state = utils.create_state_from_dict(state_dict)
                goal = set()
                for tgt in tgts:
                    goal.add(GroundAtom(self._Measured, [tgt]))
                task = EnvironmentTask(init_state, goal)
                tasks.append(task)
                task_id += 1
                logging.info(f"Task {task_id} generated.")
            else:
                logging.warning(f"Task {task_id} skipped.")
        return tasks

    def sample_tgt_using_arm(self, rng: np.random.Generator, 
                             convex_hulls: List[utils._Geom2D],
                             tgt_x: float, tgt_y: float,
                             stair_height: float, off_stair: bool
                             ) -> Tuple[float, float, float, float, float, float, float]:
        # calculate the spot pitch and z if on stair
        body_pitch_on_stair = DEFAULT_STAIR2BODY_ONSTAIR_TF.rot.to_pitch()
        imagined_stair_pose = math_helpers.SE3Pose(x=0, y=0, z=stair_height + self.ground_z, 
                                                   rot=math_helpers.Quat())
        imagined_onstair = imagined_stair_pose.mult(DEFAULT_STAIR2BODY_ONSTAIR_TF)
        body_z_on_stair = imagined_onstair.z
        # off-stair pitch and z
        body_pitch_off_stair = 0.0
        body_z_off_stair = self.spot_height + self.ground_z
        # sample a tgt pose such that Spot has to use the stair to see the target
        sampling_time = 0
        while True:
            sampling_time += 1
            # logging.info(f"Sampling time: {sampling_time}")
            # z value and x axis direction is most important
            if off_stair:
                # spot is off stair
                # slightly face down
                tgt_z = rng.uniform(self.tgt_height_range[0], self.tgt_height_range[1])
                tgt_x_axis_z = rng.uniform(-0.01, 0.01)
            else:
                # spot is on stair
                # slightly face up
                tgt_z = rng.uniform(self.high_tgt_height_range[0], self.high_tgt_height_range[1])
                tgt_x_axis_z = rng.uniform(0.0, 0.01)
            tgt_z += self.ground_z
            negated_tgt_yaw = np.arctan2(tgt_y - (self.render_y_ub + self.render_y_lb) / 2, \
                                 tgt_x - (self.render_x_ub + self.render_x_lb) / 2) # tgt_x axis in world frame pointing to origin
            angle_noise = rng.uniform(-np.pi/36, np.pi/36)
            negated_tgt_yaw += angle_noise
            # use z and yaw to determine tgt x axis
            tgt_x_axis_xy = np.sqrt(1 - tgt_x_axis_z**2)
            tgt_x_axis_x = -tgt_x_axis_xy * np.cos(negated_tgt_yaw)
            tgt_x_axis_y = -tgt_x_axis_xy * np.sin(negated_tgt_yaw)
            tgt_x_axis = np.array([tgt_x_axis_x, tgt_x_axis_y, tgt_x_axis_z])
            # translate to origin
            tgt_position = np.array([tgt_x, tgt_y, tgt_z])

            # get desired hand pose
            hand_tgt_dist = (CFG.cam2obj_distance_tol[0] + CFG.cam2obj_distance_tol[1]) / 2
            hand_position = tgt_position + hand_tgt_dist * tgt_x_axis
            hand_x_axis = -tgt_x_axis
            world_z_axis = np.array([0.0, 0.0, 1.0])
            tgt_z_axis = world_z_axis - np.dot(world_z_axis, hand_x_axis) * hand_x_axis
            hand_z_axis = tgt_z_axis / np.linalg.norm(tgt_z_axis)
            hand_y_axis = np.cross(hand_z_axis, hand_x_axis)
            hand_rot = math_helpers.Quat.from_matrix(np.array([hand_x_axis, hand_y_axis, hand_z_axis]).T)
            hand_pose = math_helpers.SE3Pose(x=hand_position[0], y=hand_position[1], z=hand_position[2], rot=hand_rot)
            # assume tgt has the same z axis as hand, the x and y axis are in reversed direction
            tgt_rot = math_helpers.Quat.from_matrix(np.array([tgt_x_axis, -hand_y_axis, hand_z_axis]).T)
            tgt_pose = math_helpers.SE3Pose(x=tgt_position[0], y=tgt_position[1], z=tgt_position[2], rot=tgt_rot)
            hand_pose_mat = hand_pose.to_matrix()
            tgt_yaw = tgt_pose.rot.to_yaw()
            pointing_theta = tgt_yaw + np.pi if tgt_yaw < 0 else tgt_yaw - np.pi
            dx_max = CFG.viewobj_distance[1] * np.cos(tgt_yaw)
            dx_min = CFG.viewobj_distance[0] * np.cos(tgt_yaw)
            dy_max = CFG.viewobj_distance[1] * np.sin(tgt_yaw)
            dy_min = CFG.viewobj_distance[0] * np.sin(tgt_yaw)
            dx = [dx_min, dx_max]
            dy = [dy_min, dy_max]
            dtheta = CFG.move2hand_view_yaw_tol
            
            # specify body pose constraints on/off stair
            body_constraint_off_stair = [
                [tgt_x + min(dx), tgt_x + max(dx)],
                [tgt_y + min(dy), tgt_y + max(dy)],
                [body_z_off_stair - 0.001, body_z_off_stair + 0.001], # z
                [-0.0001, 0.0001], # roll
                [body_pitch_off_stair - 0.0001, body_pitch_off_stair + 0.0001], # pitch
                [pointing_theta - dtheta, pointing_theta + dtheta], # yaw  
            ]
            off_stair_reachable, sol = self.spot_kinematics.compute_whole_body_ik(hand_pose_mat, body_constraint_off_stair)
            if off_stair:
                # we want off_stair_reachable
                if off_stair_reachable:
                    body_pose_xyz = sol[:3]
                    body_rot_angle = sol[3:6]
                    cand_geom = Rectangle.from_center(body_pose_xyz[0], body_pose_xyz[1], 0.85,
                                            0.25, body_rot_angle[2])
                    if valid_navigation_position(
                        cand_geom, set(), convex_hulls):
                        return True, tgt_z, tgt_pose.rot.w, tgt_pose.rot.x, tgt_pose.rot.y, tgt_pose.rot.z
                    else:
                        return False, 0, 0, 0, 0, 0
                else:
                    continue
            else:
                # we do not want off_stair_reachable
                if off_stair_reachable:
                    continue
            # Keep consistent with sampler
            dx_max = abs(CFG.put_stair_tgt_distance * np.cos(tgt_yaw))
            # dx_min = CFG.put_stair_tgt_distance[0] * np.cos(tgt_yaw)
            dy_max = abs(CFG.put_stair_tgt_distance * np.sin(tgt_yaw))
            # dy_min = CFG.put_stair_tgt_distance[0] * np.sin(tgt_yaw)
            dx = [-dx_max, dx_max]
            dy = [-dy_max, dy_max]
            dtheta = CFG.put_stair_tgt_yaw_tol
            body_constraint_on_stair = [
                [tgt_x + min(dx), tgt_x + max(dx)],
                [tgt_y + min(dy), tgt_y + max(dy)],
                [body_z_on_stair - 0.001, body_z_on_stair + 0.001], # z
                [-0.0001, 0.0001], # roll
                [body_pitch_on_stair - 0.0001, body_pitch_on_stair + 0.0001], # pitch
                [pointing_theta - dtheta, pointing_theta + dtheta], # yaw  
            ]
            on_stair_reachable, sol = self.spot_kinematics.compute_whole_body_ik(hand_pose_mat, body_constraint_on_stair)
            if on_stair_reachable:
                # Note that we only checked onstair valid here
                # Assuming offstair is valid
                body_pose_xyz = sol[:3]
                body_rot_angle = sol[3:6]
                cand_geom = Rectangle.from_center(body_pose_xyz[0], body_pose_xyz[1], 0.85,
                                        0.25, body_rot_angle[2])
                if valid_navigation_position(
                    cand_geom, set(), convex_hulls):
                    return True, tgt_z, tgt_pose.rot.w, tgt_pose.rot.x, tgt_pose.rot.y, tgt_pose.rot.z
                else:
                    return False, 0, 0, 0, 0, 0
            else:
                continue
    
    def sample_stair_using_arm(self, rng: np.random.Generator, 
                             convex_hulls: Set[utils._Geom2D],
                             stair_x: float, stair_y: float,
                             stair_height: float
                             ) -> Tuple[float, float, float, float]:
        # sample a tgt pose such that Spot has to use the stair to see the target
        sampling_time = 0
        body_z_off_stair = self.spot_height + self.ground_z
        stair_z = stair_height + self.ground_z
        body_pitch_off_stair = 0.0
        while True:
            sampling_time += 1
            # logging.info(f"Sampling time: {sampling_time}")
            # z value and x axis direction is most important
            # tgt_x axis in world frame pointing to origin
            base_yaw = np.arctan2(stair_y - (self.render_y_ub + self.render_y_lb) / 2, \
                                  stair_x - (self.render_x_ub + self.render_x_lb) / 2) - np.pi
            angle_noise = rng.uniform(-np.pi/6, np.pi/6)
            base_yaw += angle_noise
            # use z and yaw to determine tgt x axis
            stair_rot = math_helpers.Quat.from_yaw(base_yaw)
            stair_pose = math_helpers.SE3Pose(x=stair_x, y=stair_y, z=stair_z, rot=stair_rot)
            hand_pose = stair_pose.mult(DEFAULT_STAIR2HAND_TF)
            hand_pose_mat = hand_pose.to_matrix()
            
            # specify body pose constraints on/off stair
            dx_max = CFG.reach_stair_distance[1] * np.cos(base_yaw)
            dx_min = CFG.reach_stair_distance[0] * np.cos(base_yaw)
            dy_max = CFG.reach_stair_distance[1] * np.sin(base_yaw)
            dy_min = CFG.reach_stair_distance[0] * np.sin(base_yaw)
            body_yaw = base_yaw + np.pi if base_yaw < 0 else base_yaw - np.pi
            body_constraint_off_stair = [
                [stair_x + min(dx_min, dx_max), stair_x + max(dx_min, dx_max)], # x
                [stair_y + min(dy_min, dy_max), stair_y + max(dy_min, dy_max)], # y
                [body_z_off_stair - 0.001, body_z_off_stair + 0.001], # z
                [-0.0001, 0.0001], # roll
                [-0.0001, 0.0001], # pitch
                [body_yaw - CFG.reach_stair_yaw_tol, body_yaw + CFG.reach_stair_yaw_tol], # yaw
            ]
            off_stair_reachable, sol = self.spot_kinematics.compute_whole_body_ik(hand_pose_mat, body_constraint_off_stair)
            if off_stair_reachable:
                body_pose_xyz = sol[:3]
                body_rot_angle = sol[3:6]
                cand_geom = Rectangle.from_center(body_pose_xyz[0], body_pose_xyz[1], 0.85,
                                        0.25, body_rot_angle[2])
                if valid_navigation_position(
                    cand_geom, set(), convex_hulls):
                    return True, stair_pose.rot.x, stair_pose.rot.y, stair_pose.rot.z, stair_pose.rot.w
                else:
                    return False, 0, 0, 0, 0
            else:
                continue

    def render_state_plt(self,
                        state: State,
                        task: EnvironmentTask,
                        action: Optional[Action] = None,
                        caption: Optional[str] = None,
                        save_path: Optional[str] = None):
        
        # Set the image resolution
        width, height = 1490, 1080  # Modify as needed

        # Create an offscreen renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height) # type: ignore
        
        # Set up scene
        scene = renderer.scene
        
        # World coordinates (x, y, z axes)
        world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])
        scene.add_geometry("world_frame", world_coordinate_frame, o3d.visualization.rendering.MaterialRecord()) # type: ignore
        
        # Load the RGB image
        plane = o3d.geometry.TriangleMesh.create_box(width=(self.render_x_ub - self.render_x_lb), \
                                                    height=(self.render_y_ub - self.render_y_lb), \
                                                    depth=0.01)
        plane_translation = np.array([self.render_x_lb, self.render_y_lb, self.ground_z])
        # Optional: Assign a material to the box (if you want to color it, etc.)
        ground_material = o3d.visualization.rendering.MaterialRecord() # type: ignore
        ground_material.base_color = [0.55, 0.27, 0.07, 0.8]  # Set the box color (RGBA)

        # Add the plane to the scene with the material
        plane.translate(plane_translation)
        scene.add_geometry("plane", plane, ground_material)

        # Robot body frame (spot_robot)
        spot_robot = state.get_objects(self._spot_type)[0]
        
        spot_x = state.get(spot_robot, "x")
        spot_y = state.get(spot_robot, "y")
        spot_z = state.get(spot_robot, "z")
        spot_qx = state.get(spot_robot, "qx")
        spot_qy = state.get(spot_robot, "qy")
        spot_qz = state.get(spot_robot, "qz")
        spot_qw = state.get(spot_robot, "qw")
        
        # Convert quaternion to rotation matrix
        spot_rotation = R.from_quat([spot_qx, spot_qy, spot_qz, spot_qw]).as_matrix()
        spot_translation = np.array([spot_x, spot_y, spot_z])
        
        # Create robot body frame (as a coordinate frame)
        robot_body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        robot_body_frame.translate(spot_translation)
        robot_body_frame.rotate(spot_rotation, center=spot_translation)
        scene.add_geometry("robot_body_frame", robot_body_frame, o3d.visualization.rendering.MaterialRecord()) # type: ignore

        # Define the box representing the robot (you can adjust size as needed)
        box_size = [0.85, 0.25, 0.2]  # Length, Width, Height of Spot Back
        robot_box = o3d.geometry.TriangleMesh.create_box(width=box_size[0], height=box_size[1], depth=box_size[2])

        # Translate the box so that the top plane is aligned with the XoY plane of the coordinate frame
        # This means the box's Z center will be shifted by half the box's height in the negative Z direction
        box_translation = np.array([spot_x, spot_y, 
                                    spot_z])

        # Apply a further translation to account for the box being created with its origin at a corner
        # We plot the arm base pose, which we assume it is located at 1/6 of the box length
        box_center_shift = np.array([box_size[0] * 1 / 2, box_size[1] / 2, box_size[2] / 2])

        # Translate the box to the correct position
        robot_box.translate(box_translation - box_center_shift)
        robot_box.rotate(spot_rotation, center=box_translation)

        # Optional: Assign a material to the box (if you want to color it, etc.)
        box_material = o3d.visualization.rendering.MaterialRecord() # type: ignore
        box_material.base_color = [1.0, 1.0, 0.1, 1.0]  # Set the box color (RGBA)

        # Add the box to the scene
        scene.add_geometry("robot_box", robot_box, box_material)
        
        # Robot hand (camera) frame
        hand_x = state.get(spot_robot, "ee_x")
        hand_y = state.get(spot_robot, "ee_y")
        hand_z = state.get(spot_robot, "ee_z")
        hand_qx = state.get(spot_robot, "ee_qx")
        hand_qy = state.get(spot_robot, "ee_qy")
        hand_qz = state.get(spot_robot, "ee_qz")
        hand_qw = state.get(spot_robot, "ee_qw")
        robot_calibrated = state.get(spot_robot, "calibrated")
        
        hand_rotation = R.from_quat([hand_qx, hand_qy, hand_qz, hand_qw]).as_matrix()
        hand_translation = np.array([hand_x, hand_y, hand_z])
        # Create camera frame (as another coordinate frame)
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        camera_frame.translate(hand_translation)
        camera_frame.rotate(hand_rotation, center=hand_translation)
        scene.add_geometry("camera_frame", camera_frame, o3d.visualization.rendering.MaterialRecord()) # type: ignore

        # add the targets
        name2info = {}
        for tgt in state.get_objects(self._tgt_obj_type):
            tgt_x = state.get(tgt, "x")
            tgt_y = state.get(tgt, "y")
            tgt_z = state.get(tgt, "z")
            tgt_qx = state.get(tgt, "qx")
            tgt_qy = state.get(tgt, "qy")
            tgt_qz = state.get(tgt, "qz")
            tgt_qw = state.get(tgt, "qw")
            tgt_r = state.get(tgt, "r")
            tgt_g = state.get(tgt, "g")
            tgt_b = state.get(tgt, "b")
            tgt_rotation = R.from_quat([tgt_qx, tgt_qy, tgt_qz, tgt_qw]).as_matrix()
            tgt_translation = np.array([tgt_x, tgt_y, tgt_z])
            tgt_sz = state.get(tgt, "height")
            tgt_geom = o3d.geometry.TriangleMesh.create_box(width=tgt_sz, height=tgt_sz, depth=tgt_sz)
            tgt_shift = np.array([tgt_sz / 2, tgt_sz / 2, tgt_sz / 2])
            tgt_geom.translate(tgt_translation - tgt_shift)
            tgt_geom.rotate(tgt_rotation, center=tgt_translation)

            tgt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            tgt_frame.translate(tgt_translation)
            tgt_frame.rotate(tgt_rotation, center=tgt_translation)
            scene.add_geometry(f"{tgt.name}_frame", tgt_frame, o3d.visualization.rendering.MaterialRecord()) # type: ignore
            # add color to the target cube
            tgt_material = o3d.visualization.rendering.MaterialRecord() # type: ignore
            assert tgt.name not in name2info
            color = np.array([tgt_r, tgt_g, tgt_b]) / np.array([tgt_r, tgt_g, tgt_b]).sum()
            name2info[tgt.name] = {
                'color': list(color),
                'measured': state.get(tgt, "measured")
            }
            tgt_material.base_color = list(color) + [1.0]  # Set the box color (RGBA)
            scene.add_geometry(tgt.name, tgt_geom, tgt_material)

        for stair in state.get_objects(self._stairs_type):
            tgt_x = state.get(stair, "x")
            tgt_y = state.get(stair, "y")
            tgt_z = state.get(stair, "z")
            tgt_qx = state.get(stair, "qx")
            tgt_qy = state.get(stair, "qy")
            tgt_qz = state.get(stair, "qz")
            tgt_qw = state.get(stair, "qw")
            tgt_r = state.get(stair, "r")
            tgt_g = state.get(stair, "g")
            tgt_b = state.get(stair, "b")
            tgt_rotation = R.from_quat([tgt_qx, tgt_qy, tgt_qz, tgt_qw]).as_matrix()
            tgt_translation = np.array([tgt_x, tgt_y, tgt_z])
            tgt_sz = state.get(stair, "width")
            tgt_height = state.get(stair, "height")
            tgt_geom = o3d.geometry.TriangleMesh.create_box(width=tgt_sz, height=tgt_sz, depth=tgt_height)
            tgt_shift = np.array([tgt_sz / 2, tgt_sz / 2, tgt_height])
            tgt_geom.translate(tgt_translation - tgt_shift)
            tgt_geom.rotate(tgt_rotation, center=tgt_translation)

            tgt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            tgt_frame.translate(tgt_translation)
            tgt_frame.rotate(tgt_rotation, center=tgt_translation)
            scene.add_geometry(f"{stair.name}_frame", tgt_frame, o3d.visualization.rendering.MaterialRecord()) # type: ignore
            # add color to the target cube
            tgt_material = o3d.visualization.rendering.MaterialRecord() # type: ignore
            assert stair.name not in name2info
            color = np.array([tgt_r, tgt_g, tgt_b]) / np.array([tgt_r, tgt_g, tgt_b]).sum()
            name2info[stair.name] = {
                'color': list(color),
            }
            tgt_material.base_color = list(color) + [1.0]  # Set the box color (RGBA)
            scene.add_geometry(stair.name, tgt_geom, tgt_material)

        # finally visualize the allowed region
        metadata = load_spot_metadata()
        allowed_regions = metadata.get("allowed-regions", {})
        # Call the function to draw convex hulls
        for i, convex_hull in enumerate(allowed_regions.values()):
            convex_hull = np.array(convex_hull)
            points_2d = convex_hull[:, :2]
            region_mesh = o3d.geometry.TriangleMesh()
            for pt in points_2d:
                region_mesh.vertices.append([pt[0], pt[1], self.ground_z + 0.01])
            triangles = []
            for j in range(1, len(points_2d) - 1):
                triangles.append([0, j, j + 1])
            region_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            region_material = o3d.visualization.rendering.MaterialRecord()
            region_material.base_color = [0.0, 1.0, 0.0, 0.2]  # Green with some transparency
            scene.add_geometry(f"allowed_region_{i}", region_mesh, region_material)
        
        # optionally visualize the joints
        if self.visualize_joints:
            # add the joints between body and arm
            body_pose_mat = math_helpers.SE3Pose(x=spot_x, y=spot_y, z=spot_z, 
                            rot=math_helpers.Quat.from_matrix(spot_rotation)).to_matrix()
            hand_pose_mat = math_helpers.SE3Pose(x=hand_x, y=hand_y, z=hand_z, 
                            rot=math_helpers.Quat.from_matrix(hand_rotation)).to_matrix()
            succ, joints = self.spot_kinematics.compute_ik(body_pose_mat, hand_pose_mat)
            if not succ:
                logging.warning("IK failed for current state")
                joints = DEFAULT_HAND_STOW_ANGLES
            joint_poses = self.spot_kinematics.compute_fk(body_pose_mat, joints)
            for i, joint_pose in enumerate(joint_poses):
                if i == 0:
                    continue
                joint_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                joint_geom.transform(joint_pose)
                scene.add_geometry(f"joint{i}_frame", joint_geom, o3d.visualization.rendering.MaterialRecord()) # type: ignore
                if i < len(joint_poses) - 1:
                    start = joint_pose[:3, 3]
                    end = joint_poses[i + 1][:3, 3]
                    if np.linalg.norm(start - end) < 0.01:
                        continue
                    # Compute the cylinder properties
                    cylinder_radius = 0.03  # Adjust thickness
                    cylinder_height = np.linalg.norm(end - start)
                    cylinder_geom = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=cylinder_height)
                    
                    # Align the cylinder between start and end points
                    mid_point = (start + end) / 2.0
                    direction = (end - start) / cylinder_height
                    rotation_matrix = np.eye(4)
                    rotation_matrix[:3, :3] = np.linalg.svd(np.cross(np.eye(3), direction), full_matrices=False)[2].T
                    rotation_matrix[:3, 3] = mid_point
                    
                    cylinder_geom.transform(rotation_matrix)

                    # Add material to the cylinder
                    material = o3d.visualization.rendering.MaterialRecord() # type: ignore
                    material.shader = "defaultLit"  # Use lit shader for solid geometry
                    scene.add_geometry(f"link{i}", cylinder_geom, material)

        # List of different camera positions (you can add more views here)
        cam_x = (self.render_x_ub + self.render_x_lb) / 2
        cam_y = (self.render_y_ub + self.render_y_lb) / 2
        camera_positions = [
            [-cam_x+self.render_x_lb-2, cam_y, 0.4], 
            [cam_x, cam_y, 6], 
            [-cam_x + self.render_x_lb, -cam_y + self.render_y_lb, 0.4],
            [cam_x, -cam_y + self.render_y_lb-1, 0.4],  
        ]

        up_directions = [
            [0.05, 0, 0.95],   # Up direction for view 3
            [0, 1, 0],  # Up direction for view 1
            [0.05, 0.05, 0.95],
            [0, 0.05, 0.95],  # Up direction for view 2
        ]
        
        # Render multiple views and store images
        rendered_images = []  # Store rendered images here

        for i, camera_position in enumerate(camera_positions):
            # Set camera view
            camera = renderer.scene.camera
            camera.look_at(np.array([cam_x, cam_y, self.ground_z]), 
                           camera_position, up_directions[i])
            
            # Render the scene and capture the image
            image = renderer.render_to_image()
            
            # Convert Open3D image to a NumPy array (RGBA)
            image_np = np.asarray(image)
            
            # Store the NumPy image
            rendered_images.append(image_np)

        # Display the images in a single Matplotlib figure
        fig, axs = plt.subplots(2, 2, figsize=(90, 50))  # 2 row, 2 columns

        n = 0
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                # Display each image in a subplot
                ax.imshow(rendered_images[n])
                ax.axis('off')
                ax.set_title(f"View {n+1}")
                n += 1
    
        # Adjust the layout to provide more space at the bottom
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.98, bottom=0.25, left=0.01, right=0.99)

        # Create a new axis for the caption (position it at the bottom)
        caption_ax = fig.add_axes([0.3, 0.01, 0.9, 0.1])  # Position at the bottom within the figure bounds
        caption_ax.axis('off')  # Hide the axis

        # Create a list of colored squares (patches) and text for the caption
        patches = []
        labels = []
        # first add robot
        color_patch = mpatches.Rectangle((0, 0), 30, 30, facecolor=[1.0, 1.0, 0.2], 
                                         edgecolor="none")
        patches.append(color_patch)
        labels.append(f"Spot - C: {robot_calibrated}")
        for name, info in name2info.items():
            # Create a small colored square
            color_patch = mpatches.Rectangle((0, 0), 30, 30, facecolor=info['color'], edgecolor="none")
            patches.append(color_patch)
            if 'tgt' in name:
                labels.append(f"{name} - M: {info['measured']}")
            else:
                labels.append(f"{name}")

        # Add the legend below the figure
        caption_ax.legend(patches, labels, loc='center', fontsize=50, frameon=False, ncol=len(name2info))

        # Optionally apply tight_layout as an additional adjustment
        # plt.tight_layout(rect=[0, 0, 0.1, 0.4])
        if save_path:
            fig.savefig(save_path, format='png', dpi=40)
        
        return fig
    
    def _DirectViewable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        tgt = objects[0]
        #  get the stair id
        tgt_stair_id = state.get(tgt, "stair_id")
        if tgt_stair_id == -1:
            return True
        return False

    def _AppliedTo_holds(self, state: State, objects: Sequence[Object]) -> bool:
        stair, tgt = objects
        #  get the stair id
        stair_id = state.get(stair, "object_id")
        tgt_stair_id = state.get(tgt, "stair_id")
        if tgt_stair_id == stair_id:
            return True
        return False
    
    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, stair = objects
        #  get the x y z
        robot_x = state.get(robot, "ee_x")
        robot_y = state.get(robot, "ee_y")
        robot_z = state.get(robot, "ee_z")

        stair_x = state.get(stair, "x")
        stair_y = state.get(stair, "y")
        stair_z = state.get(stair, "z")
        if (robot_x - stair_x)**2 + (robot_y - stair_y)**2 + (robot_z - stair_z)**2 < 0.1:
            return True
        return False
    
    def _Near_holds(self, state: State, objects: Sequence[Object]) -> bool:
        stair, tgt = objects
        #  check if the spot is on stair, it can hand see the target
        stair_pose = utils.get_se3_pose_from_state(state, stair)
        # foot pose on stair
        body_pose_on_stair = stair_pose * DEFAULT_STAIR2BODY_ONSTAIR_TF
        body_pose_on_stair_mat = body_pose_on_stair.to_matrix()
        # hand pose
        obj_pose = utils.get_se3_pose_from_state(state, tgt)
        tgt_x_axis = obj_pose.to_matrix()[:3, 0]
        tgt_z_axis = obj_pose.to_matrix()[:3, 2]
        tgt_position = np.array([obj_pose.x, 
                                obj_pose.y,
                                obj_pose.z])
        hand_tgt_dist = (CFG.cam2obj_distance_tol[0] + CFG.cam2obj_distance_tol[1]) / 2
        hand_position = tgt_position + hand_tgt_dist * tgt_x_axis
        hand_x_axis = -tgt_x_axis
        hand_z_axis = tgt_z_axis
        hand_y_axis = np.cross(hand_z_axis, hand_x_axis)
        hand_rot = math_helpers.Quat.from_matrix(np.array([hand_x_axis, hand_y_axis, hand_z_axis]).T)
        hand_pose = math_helpers.SE3Pose(x=hand_position[0], y=hand_position[1], z=hand_position[2], rot=hand_rot)
        hand_pose_mat = hand_pose.to_matrix()
        # Check if IK has a solution.
        robot = SpotArmFK()
        suc, _ = robot.compute_ik(body_pose_on_stair_mat, hand_pose_mat)
        if suc:
            return True
        return False
    
    def _Close_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, tgt = objects
        #  get the x y
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")

        tgt_x = state.get(tgt, "x")
        tgt_y = state.get(tgt, "y")

        dist_2d = np.sqrt((robot_x - tgt_x)**2 + (robot_y - tgt_y)**2)
        # simply use distance
        if dist_2d < CFG.put_stair_tgt_distance:
            return True
        return False
    
    def _Measured_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj = objects[0]
        return bool(state.get(obj, "measured"))
    
    def _CalibrationTgt_holds(self, state: State, objects: Sequence[Object]) -> bool:
        spot = objects[0]
        tgt = objects[1]
        tgt_id = state.get(tgt, "object_id")
        spot_cal_id = state.get(spot, "calibration_obj_id")
        return spot_cal_id == tgt_id
    
    def _Calibrated_holds(self, state: State, objects: Sequence[Object]) -> bool:
        spot = objects[0]
        return bool(state.get(spot, "calibrated"))
    
    def _ViewClear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # This is required otherwise Viewable will be remembered
        spot = objects[0]
        spot_pose = utils.get_se3_pose_from_state(state, spot)
        spot_x = spot_pose.x
        spot_y = spot_pose.y
        for tgt in state.get_objects(self._tgt_obj_type):
            # for faster computation
            if self._Viewable_Arm.holds(state, [spot, tgt]):
                return False
            # Can't use distance, as the workspace is hard to define
            # tgt_pose = utils.get_se3_pose_from_state(state, tgt)
            # tgt_x = tgt_pose.x
            # tgt_y = tgt_pose.y
            # dist_2d = np.sqrt((spot_x - tgt_x)**2 + (spot_y - tgt_y)**2)
            # if dist_2d < CFG.far_distance[0]:
            #     return False
        return True

    def _SurroundingClear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        spot = objects[0]
        spot_pose = utils.get_se3_pose_from_state(state, spot)
        spot_x = spot_pose.x
        spot_y = spot_pose.y
        for tgt in state.get_objects(self._stairs_type):
            # for fast computation
            # can't use reachable, as if the spot is on stair, it is also not reachable
            # But in this case, it shoud not be surrounding clear
            # if self._Reachable.holds(state, [spot, tgt]):
            #     return False
            tgt_pose = utils.get_se3_pose_from_state(state, tgt)
            tgt_x = tgt_pose.x
            tgt_y = tgt_pose.y
            dist_2d = np.sqrt((spot_x - tgt_x)**2 + (spot_y - tgt_y)**2)
            # MoveAway definitely delete it
            if dist_2d < CFG.far_distance[0]:
                return False
        return True

    def _OnGround_holds(self, state: State, objects: Sequence[Object]) -> bool:
        spot = objects[0]
        return state.get(spot, "z") <= self.spot_height + \
                                        self.ground_z + \
                                        2 * CFG.viewplan_trans_noise
    
    def _OnStair_holds(self, state: State, objects: Sequence[Object]) -> bool:
        spot, stair = objects
        stair_pose = utils.get_se3_pose_from_state(state, stair)
        imagined_spot_pose = stair_pose.mult(DEFAULT_STAIR2BODY_ONSTAIR_TF)
        spot_not_on_ground = (state.get(spot, "z") - imagined_spot_pose.z) < \
                                2 * CFG.viewplan_trans_noise
        
        spot_pose = utils.get_se3_pose_from_state(state, spot)
        imagined_stair_pose = spot_pose.mult(DEFAULT_STAIR2BODY_ONSTAIR_TF.inverse())
        dist = np.sqrt((stair_pose.x - imagined_stair_pose.x)**2 + \
                        (stair_pose.y - imagined_stair_pose.y)**2)
        return spot_not_on_ground and (dist < 2 * CFG.viewplan_trans_noise)

    def simulate(self, state: State, action: Action) -> State:
        assert isinstance(action.extra_info, (list, tuple))
        action_name, action_objs, _, action_args = action.extra_info
        next_state = state.copy()
        # Note: This is a simulated env, not spot env, so we don't need to
        # care about "Obs" vs "State" here.
        if action_name == "MoveToHandViewObject" or \
            action_name == "MoveAwayFromObject" or \
            action_name == "MoveAwayFromObjectStair" or \
            action_name == "MoveToReachObject" or \
            action_name == "MoveToPlaceObject":
            spot = action_objs[0]
            robot_rel_se2_pose = action_args[1]
            spot_pose = utils.get_se3_pose_from_state(next_state, spot)
            spot_se2_pose = spot_pose.get_closest_se2_transform()
            new_spot_se2_pose = spot_se2_pose * robot_rel_se2_pose
            new_spot_pose = new_spot_se2_pose.get_closest_se3_transform()
            # the new z of these move actions are always on ground
            new_spot_pose.z = self.spot_height + self.ground_z
            new_spot_pose = add_noise_to_pose(new_spot_pose, self._train_rng)
            # the new z axis is always pointing up
            next_state.set(spot, "x", new_spot_pose.x)
            next_state.set(spot, "y", new_spot_pose.y)
            next_state.set(spot, "z", new_spot_pose.z)
            next_state.set(spot, "qx", new_spot_pose.rot.x)
            next_state.set(spot, "qy", new_spot_pose.rot.y)
            next_state.set(spot, "qz", new_spot_pose.rot.z)
            next_state.set(spot, "qw", new_spot_pose.rot.w)
            # update the EE pose
            if 'Away' in action_name:
                # MoveAway always firs stow arm
                new_hand_pose_mat = self.spot_kinematics.compute_fk(new_spot_pose.to_matrix(), \
                                    DEFAULT_HAND_STOW_ANGLES)[-1]
                new_hand_pose = math_helpers.SE3Pose.from_matrix(new_hand_pose_mat)
                new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            else:
                # Otherwise, assme hand is in fixed rel pose to body
                hand_pose = utils.get_se3_hand_pose_from_state(next_state, spot)
                # Assume the hand pose is relatively fixed to the spot pose
                new_hand_pose = new_spot_pose.mult(spot_pose.inverse().\
                                                mult(hand_pose))
                new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
            # update stair pose if in hand
            stair_objs = state.get_objects(self._stairs_type)
            for stair_obj in stair_objs:
                if self._Holding_holds(state, [spot, stair_obj]):
                    # Assume the stair pose is relatively fixed to the spot pose
                    new_stair_pose = new_hand_pose.mult(DEFAULT_HAND2STAIR_TF)
                    next_state.set(stair_obj, "x", new_stair_pose.x)
                    next_state.set(stair_obj, "y", new_stair_pose.y)
                    next_state.set(stair_obj, "z", new_stair_pose.z)
                    next_state.set(stair_obj, "qx", new_stair_pose.rot.x)
                    next_state.set(stair_obj, "qy", new_stair_pose.rot.y)
                    next_state.set(stair_obj, "qz", new_stair_pose.rot.z)
                    next_state.set(stair_obj, "qw", new_stair_pose.rot.w)
                    break
            return next_state
        elif action_name == "MoveToOnStairsHandViewObject":
            # note that we actually didn't use the sampler in the simulation
            # assume the robot is stepping on the center of the stair
            spot, stair, tgt = action_objs[0], action_objs[1], action_objs[2]
            if not self._AppliedTo_holds(next_state, [stair, tgt]):
                return next_state
            if not self._Near_holds(next_state, [stair, tgt]):
                return next_state
            # first get the stair pose, which is the foot pose
            stair_pose = utils.get_se3_pose_from_state(next_state, stair)
            body_pose_on_stair = stair_pose.mult(DEFAULT_STAIR2BODY_ONSTAIR_TF)
            body_pose_on_stair = add_noise_to_pose(body_pose_on_stair, self._train_rng)
            # the new z axis is always pointing up
            next_state.set(spot, "x", body_pose_on_stair.x)
            next_state.set(spot, "y", body_pose_on_stair.y)
            next_state.set(spot, "z", body_pose_on_stair.z)
            next_state.set(spot, "qx", body_pose_on_stair.rot.x)
            next_state.set(spot, "qy", body_pose_on_stair.rot.y)
            next_state.set(spot, "qz", body_pose_on_stair.rot.z)
            next_state.set(spot, "qw", body_pose_on_stair.rot.w)
            # update the EE pose
            new_hand_pose_mat = self.spot_kinematics.compute_fk(body_pose_on_stair.to_matrix(), \
                                DEFAULT_HAND_STOW_ANGLES)[-1]
            new_hand_pose = math_helpers.SE3Pose.from_matrix(new_hand_pose_mat)
            new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
        elif action_name == "PickObjectFromTop":
            spot, stair = action_objs
            if not self._HandEmpty.holds(next_state, [spot]):
                return next_state
            if not self._Reachable.holds(next_state, [spot, stair]):
                return next_state
            # note that we actually didn't use the sampler in the simulation
            # assume the robot directly grasping the stair's top handle
            # and the stair is teleported to the spot's hand
            # First, move to hand to default holding pose
            body_pose = utils.get_se3_pose_from_state(next_state, spot)
            body_pose = add_noise_to_pose(body_pose, self._train_rng)
            hand_pose_mat = self.spot_kinematics.compute_fk(body_pose.to_matrix(), \
                                    DEFAULT_HAND_HOLDING_ANGLES)[-1]
            new_hand_pose = math_helpers.SE3Pose.from_matrix(hand_pose_mat)
            new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            hand2obj_tf = DEFAULT_HAND2STAIR_TF
            new_stair_pose = new_hand_pose.mult(hand2obj_tf)
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
            next_state.set(spot, "gripper_open_percentage", 0.0)
            next_state.set(stair, "x", new_stair_pose.x)
            next_state.set(stair, "y", new_stair_pose.y)
            next_state.set(stair, "z", new_stair_pose.z)
            next_state.set(stair, "qx", new_stair_pose.rot.x)
            next_state.set(stair, "qy", new_stair_pose.rot.y)
            next_state.set(stair, "qz", new_stair_pose.rot.z)
            next_state.set(stair, "qw", new_stair_pose.rot.w)
            # update the "held" attribute of stair
            next_state.set(stair, "held", 1.0)
        elif action_name == "PlaceObjectInFront":
            spot, stair, _ = action_objs
            if not self._Holding_holds(next_state, [spot, stair]):
                return next_state
            # note that we actually didn't use the sampler in the simulation
            # assume the robot directly placing the stair on the ground
            # and the stair is teleported to the default offset
            spot_pose = utils.get_se3_pose_from_state(next_state, spot)
            # This is a little bit noisy, stair not flattly on ground
            dump_hand_pose = spot_pose.mult(DEFAULT_DUMPED_TF)
            new_stair_pose_noisy = dump_hand_pose.mult(DEFAULT_HAND2STAIR_TF)
            new_stair_pose_yaw = math_helpers.Quat.from_matrix(new_stair_pose_noisy.to_matrix()).to_yaw()
            new_stair_pose = math_helpers.SE3Pose(x=new_stair_pose_noisy.x, y=new_stair_pose_noisy.y, 
                                                  z=new_stair_pose_noisy.z, \
                                                rot=math_helpers.Quat.from_yaw(new_stair_pose_yaw))
            # sample arm initial pose, body 2 hand
            spot_pose = add_noise_to_pose(spot_pose, self._train_rng)
            hand_pose_mat = self.spot_kinematics.compute_fk(spot_pose.to_matrix(), \
                                            DEFAULT_HAND_STOW_ANGLES)[-1]
            hand_pose = math_helpers.SE3Pose.from_matrix(hand_pose_mat)
            hand_pose = add_noise_to_pose(hand_pose, self._train_rng)
            stair_height = next_state.get(stair, "height")
            next_state.set(spot, "gripper_open_percentage", 1.0)
            next_state.set(spot, "ee_x", hand_pose.x)
            next_state.set(spot, "ee_y", hand_pose.y)
            next_state.set(spot, "ee_z", hand_pose.z)
            next_state.set(spot, "ee_qx", hand_pose.rot.x)
            next_state.set(spot, "ee_qy", hand_pose.rot.y)
            next_state.set(spot, "ee_qz", hand_pose.rot.z)
            next_state.set(spot, "ee_qw", hand_pose.rot.w)
            next_state.set(stair, "x", new_stair_pose.x)
            next_state.set(stair, "y", new_stair_pose.y)
            next_state.set(stair, "z", self.ground_z + stair_height)
            next_state.set(stair, "qx", new_stair_pose.rot.x)
            next_state.set(stair, "qy", new_stair_pose.rot.y)
            next_state.set(stair, "qz", new_stair_pose.rot.z)
            next_state.set(stair, "qw", new_stair_pose.rot.w)
            # update the "held" attribute of stair
            next_state.set(stair, "held", 0.0)
        elif action_name == "Calibrate":
            # first check if the spots hand sees the calibration target
            spot, obj = action_objs
            if not self._HandSees.holds(next_state, [spot, obj]):
                return next_state
            # then check if the tgt is the calibration tgt
            if not self._CalibrationTgt_holds(next_state, [spot, obj]):
                return next_state
            # finally, set the calibrated flag
            next_state.set(spot, "calibrated", 1)
            return next_state
        elif action_name == "Measure":
            # first check if the spots hand sees the target
            spot, obj = action_objs
            if not self._HandSees.holds(next_state, [spot, obj]):
                return next_state
            # then check if the spot is calibrated
            if not self._Calibrated_holds(next_state, [spot]):
                return next_state
            # finally, set the measured flag
            next_state.set(obj, "measured", 1)
            return next_state
        elif action_name == "HandViewObject":
            spot = action_objs[0]
            hand_rel_se3_pose = action_args[1] #body2newhand
            # update the EE pose
            body_pose = utils.get_se3_pose_from_state(state, spot)
            # Assume the hand pose is relatively fixed to the spot pose
            new_hand_pose = body_pose * hand_rel_se3_pose
            new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
            return next_state
        else:
            raise ValueError(f"Unknown action name: {action_name}")
        return next_state


class SpotViewPlanHardEnv(ViewPlanHardEnv):
    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # SpotEnv Version of the ViewPlanHardEnv
        # 1. We have access to the Operator in an approach
        # 2. We use actual spot robot (ROS2 node) to get its real state
        # and image observation.
        # 3. We distinguish SpotObs and State here, as Obs is obtained
        # from the robot's current partial obs, while State is the whole state.
        # which might need to be remembered.
        # 4. Next Obs is obtained in the real-world by calling ROS2 nodes.
        # 5. The State has the same constant robot entity as the environment.
        assert "spot_wrapper" in CFG.approach or \
               "spot_wrapper" in CFG.approach_wrapper, \
            "Must use spot wrapper in spot envs!"
        robot, localizer, lease_client = get_robot()
        self._HandEmpty = Predicate("HandEmpty", [self._spot_type], self._HandEmpty_holds)
        self._robot = robot
        self._localizer = localizer
        self._lease_client = lease_client
        # Note that we need to include the operators in this
        # class because they're used to update the symbolic
        # parts of the state during execution.
        self._strips_operators: Set[STRIPSOperator] = set()
        self._approach_predicates: Set[Predicate] = set()
        self._current_task_goal_reached = False
        self._last_action: Optional[Action] = None

        # Create constant objects.
        self._spot_object = Object("robot", _robot_hand_type)

        # For noisy simulation in dry runs.
        self._noise_rng = np.random.default_rng(CFG.seed)

        # For object detection.
        self._allowed_regions: Collection[Delaunay] = get_allowed_map_regions()

        # Used for the move-related hacks in step().
        self._last_known_object_poses: Dict[Object, math_helpers.SE3Pose] = {}

    @classmethod
    def get_name(cls) -> str:
        return "spot_view_plan_hard"

    def create_pred_operators(self, approach):
        # Create the operators and predicates from the approach.
        # Note that the operators are created from an approach, which means
        # the operator may not use the predicates defined in the environment.
        assert hasattr(approach._base_approach, "_nsrts")
        for nsrt in approach._base_approach._nsrts: # type: ignore
            self._strips_operators.add(nsrt.op)
            for lifted_pred in list(nsrt.op.preconditions):
                if lifted_pred.predicate not in self._approach_predicates:
                    self._approach_predicates.add(lifted_pred.predicate)
            for lifted_pred in list(nsrt.op.add_effects):
                if lifted_pred.predicate not in self._approach_predicates:
                    self._approach_predicates.add(lifted_pred.predicate)
            for lifted_pred in list(nsrt.op.delete_effects):
                if lifted_pred.predicate not in self._approach_predicates:
                    self._approach_predicates.add(lifted_pred.predicate)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        # NOTE: task_idx and train_or_test ignored unless loading from JSON!
        self._current_task = self._get_dry_task(train_or_test, task_idx)
        if not CFG.spot_run_dry:
            prompt = f"Please set up {train_or_test} task {task_idx}!"
            input(prompt)
            assert self._lease_client is not None
            # Automatically retry if a retryable error is encountered.
            while True:
                try:
                    self._lease_client.take()
                    updated_task = self._actively_construct_env_task(self._current_task)
                    self._current_task = updated_task
                    break
                except RetryableRpcError as e:
                    logging.warning("WARNING: the following retryable error "
                                    f"was encountered. Trying again.\n{e}")
        self._current_observation = self._current_task.init_obs
        self._current_task_goal_reached = False
        self._last_action = None
        return self._current_task.init_obs
    
    def maybe_gen_subgoals(self, start: math_helpers.SE2Pose, goal: math_helpers.SE2Pose) \
    -> List[math_helpers.SE2Pose]:
        convex_hull = get_allowed_map_regions()
        start_x = start.x
        start_y = start.y
        goal_x = goal.x
        goal_y = goal.y
        start_point = np.array([start_x, start_y])
        goal_point = np.array([goal_x, goal_y])
        # Check if the start and goal are in the same allowed region
        start_region = None
        goal_region = None
        goal_hull = None
        for idx, region in enumerate(convex_hull):
            if region.find_simplex(start_point) >= 0:
                start_region = idx
                start_hull = region
            if region.find_simplex(goal_point) >= 0:
                goal_region = idx
                goal_hull = region
        if start_region == goal_region:
            return [goal]
        # Find a point in the shared region
        subgoal = None
        while subgoal is None:
            x, y = sample_point_in_hull(goal_hull, self._noise_rng)
            candidate = np.array([x, y])
            if start_hull.find_simplex(candidate) >= 0:
                subgoal = math_helpers.SE2Pose(x=x, y=y, angle=goal.angle)
        return [subgoal, goal]
    
    def _actively_construct_env_task(self, sim_task: EnvironmentTask) -> EnvironmentTask:
        # Have the spot walk around the environment once to construct
        # an initial observation.
        assert self._robot is not None
        assert self._localizer is not None
        # We hack this for now, directly use the fixed objects
        logging.info("Standing the robot up")
        spot_stand(self._robot)
        input("Next Spot will go home, press enter to continue")
        go_home(self._robot, self._localizer)
        # initially open griiper
        stow_arm(self._robot)
        open_gripper(self._robot)
        time.sleep(1)
        self._localizer.localize()
        real_pose = self._localizer.get_last_robot_pose()
        real_hand_pose = self._localizer.get_last_hand_pose()
        logging.info(f"Spot body localized at: {real_pose}")
        # Spot with specify where the stairs are
        sim_init = sim_task.init_obs
        sim_init = self.obs2state(sim_init)
        real_init = sim_init.copy()
        num_stairs = len(sim_init.get_objects(self._stairs_type))
        logging.info(f"Spot will specify {num_stairs} stairs's initial pose")
        all_stairs = sim_init.get_objects(self._stairs_type)
        distance = []
        curr_robot_se2 = real_pose.get_closest_se2_transform()
        if CFG.spot_indicate_stair:
            for stair in all_stairs:
                stair_pose = utils.get_se3_pose_from_state(sim_init, stair).get_closest_se2_transform()
                distance.append(np.sqrt((curr_robot_se2.x - stair_pose.x)**2 + \
                                        (curr_robot_se2.y - stair_pose.y)**2))
            sorted_stairs = [x for _, x in sorted(zip(distance, all_stairs), reverse=True)]
            for stair in sorted_stairs:
                stair_pose = utils.get_se3_pose_from_state(sim_init, stair)
                logging.info(f"Desired Stair Pose for {stair.name}: {stair_pose.get_closest_se2_transform()}")
                hand_pose = stair_pose.mult(DEFAULT_STAIR2HAND_TF)
                body_pose = hand_pose.mult(DEFAULT_DUMPED_TF.inverse())
                body_pose_se2 = body_pose.get_closest_se2_transform()
                # next_abs_goals = self.maybe_gen_subgoals(real_pose, body_pose_se2)
                next_abs_goals = [body_pose_se2]
                # for g in next_abs_goals:
                input(f"Next Spot will move to the stair {stair} pose {body_pose_se2}, press enter to continue")
                navigate_to_absolute_pose(self._robot, self._localizer, body_pose_se2)
                time.sleep(1)
                self._localizer.localize()
                real_pose = self._localizer.get_last_robot_pose()
                pose_err = np.sqrt((real_pose.x - body_pose.x)**2 + \
                                    (real_pose.y - body_pose.y)**2)
                logging.warning(f"Spot pose error: {pose_err}")
                time.sleep(1)
                input(f"Next Spot will specify the stair {stair} pose using hand")
                dump_tf = math_helpers.SE3Pose(
                x=0.8, y=0.0, z=-0.15, rot=math_helpers.Quat.from_pitch(np.pi / 2.5))
                move_hand_to_relative_pose(self._robot, dump_tf)
                time.sleep(1)
                self._localizer.localize()
                real_hand_pose = self._localizer.get_last_hand_pose()
                real_stair_pose = real_hand_pose.mult(DEFAULT_HAND2STAIR_TF)
                logging.info(f"Real Stair Pose: {real_stair_pose.get_closest_se2_transform()}")
                real_angle = real_stair_pose.get_closest_se2_transform().angle
                input(f"Continue when stair is ready")
                real_stair_x = real_stair_pose.x
                real_stair_y = real_stair_pose.y
                real_stair_z = real_stair_pose.z - 0.1
                real_stair_rot = math_helpers.Quat.from_yaw(real_angle)
                real_init.set(stair, "x", real_stair_x)
                real_init.set(stair, "y", real_stair_y)
                real_init.set(stair, "z", real_stair_z)
                real_init.set(stair, "qx", real_stair_rot.x)
                real_init.set(stair, "qy", real_stair_rot.y)
                real_init.set(stair, "qz", real_stair_rot.z)
                real_init.set(stair, "qw", real_stair_rot.w)
                stow_arm(self._robot)
                open_gripper(self._robot)
                time.sleep(1)
        self._localizer.localize()
        real_pose = self._localizer.get_last_robot_pose().get_closest_se2_transform()
        body_init_pose = utils.get_se3_pose_from_state(real_init, self._spot_object)
        body_init_pose_se2 = body_init_pose.get_closest_se2_transform()
        abs_subgoal_1 = real_pose * math_helpers.SE2Pose(x=0.0, y=-1.0, angle=0.0)
        abs_subgoal_2 = math_helpers.SE2Pose(x=2.0, y=3.8, angle=0.0)
        next_abs_goals = [
            # abs_subgoal_1,
            # abs_subgoal_2,
            body_init_pose_se2
        ]
        for g in next_abs_goals:
            input(f"Next Spot will move to the pose {g}, press enter to continue")
            navigate_to_absolute_pose(self._robot, self._localizer, body_init_pose_se2)
            time.sleep(1)
        self._localizer.localize()
        real_pose = self._localizer.get_last_robot_pose()
        real_pose_x = real_pose.x
        real_pose_y = real_pose.y
        real_pose_z = real_pose.z
        real_pose_qx = real_pose.rot.x
        real_pose_qy = real_pose.rot.y
        real_pose_qz = real_pose.rot.z
        real_pose_qw = real_pose.rot.w
        real_hand_pose = self._localizer.get_last_hand_pose()
        real_init.set(self._spot_object, "ee_x", real_hand_pose.x)
        real_init.set(self._spot_object, "ee_y", real_hand_pose.y)
        real_init.set(self._spot_object, "ee_z", real_hand_pose.z)
        real_init.set(self._spot_object, "ee_qx", real_hand_pose.rot.x)
        real_init.set(self._spot_object, "ee_qy", real_hand_pose.rot.y)
        real_init.set(self._spot_object, "ee_qz", real_hand_pose.rot.z)
        real_init.set(self._spot_object, "ee_qw", real_hand_pose.rot.w)
        real_init.set(self._spot_object, "x", real_pose_x)
        real_init.set(self._spot_object, "y", real_pose_y)
        real_init.set(self._spot_object, "z", real_pose_z)
        real_init.set(self._spot_object, "qx", real_pose_qx)
        real_init.set(self._spot_object, "qy", real_pose_qy)
        real_init.set(self._spot_object, "qz", real_pose_qz)
        real_init.set(self._spot_object, "qw", real_pose_qw)
        goal_description = sim_task.goal_description
        real_init = self.state2obs(real_init)
        real_task = EnvironmentTask(real_init, goal_description)
        return real_task
        # Save the task for future use.
        # json_objects = {o.name: o.type.name for o in objects_in_view}
        # json_objects[self._spot_object.name] = self._spot_object.type.name
        # init_json_dict = {
        #     o.name: {
        #         "x": pose.x,
        #         "y": pose.y,
        #         "z": pose.z,
        #         "qw": pose.rot.w,
        #         "qx": pose.rot.x,
        #         "qy": pose.rot.y,
        #         "qz": pose.rot.z,
        #     }
        #     for o, pose in objects_in_view.items()
        # }
        # # Add static object features.
        # metadata = load_spot_metadata()
        # static_object_features = metadata.get("static-object-features", {})
        # for obj_name, obj_feats in static_object_features.items():
        #     if obj_name in init_json_dict:
        #         init_json_dict[obj_name].update(obj_feats)
        # for obj in objects_in_view:
        #     if "lost" in obj.type.feature_names:
        #         init_json_dict[obj.name]["lost"] = 0.0
        #     if "in_hand_view" in obj.type.feature_names:
        #         init_json_dict[obj.name]["in_hand_view"] = 1.0
        #     if "in_view" in obj.type.feature_names:
        #         init_json_dict[obj.name]["in_view"] = 1.0
        #     if "held" in obj.type.feature_names:
        #         init_json_dict[obj.name]["held"] = 0.0
        # init_json_dict[self._spot_object.name] = {
        #     "gripper_open_percentage": gripper_open_percentage,
        #     "x": robot_pos.x,
        #     "y": robot_pos.y,
        #     "z": robot_pos.z,
        #     "qw": robot_pos.rot.w,
        #     "qx": robot_pos.rot.x,
        #     "qy": robot_pos.rot.y,
        #     "qz": robot_pos.rot.z,
        # }
        # json_dict = {
        #     "objects": json_objects,
        #     "init": init_json_dict,
        #     "goal_description": goal_description,
        # }
        # outfile = utils.get_env_asset_path("task_jsons/spot/last.json",
        #                                    assert_exists=False)
        # outpath = Path(outfile)
        # outpath.parent.mkdir(parents=True, exist_ok=True)
        # with open(outpath, "w", encoding="utf-8") as f:
        #     json.dump(json_dict, f, indent=4)
        # logging.info(f"Dumped task to {outfile}. Rename it to save it.")

    def simulate(self, state: State, action: Action) -> State:
        assert isinstance(action.extra_info, (list, tuple))
        action_name, action_objs, _, action_args = action.extra_info
        next_state = state.copy()
        # Note: This is a simulated env, not spot env, so we don't need to
        # care about "Obs" vs "State" here.
        if action_name == "MoveToHandViewObject" or \
            action_name == "MoveAwayFromObject" or \
            action_name == "MoveAwayFromObjectStair" or \
            action_name == "MoveToReachObject" or \
            action_name == "MoveToPlaceObject":
            spot = action_objs[0]
            robot_rel_se2_pose = action_args[1]
            spot_pose = utils.get_se3_pose_from_state(next_state, spot)
            spot_se2_pose = spot_pose.get_closest_se2_transform()
            new_spot_se2_pose = spot_se2_pose * robot_rel_se2_pose
            new_spot_pose = new_spot_se2_pose.get_closest_se3_transform()
            # the new z of these move actions are always on ground
            new_spot_pose.z = self.spot_height + self.ground_z
            # new_spot_pose = add_noise_to_pose(new_spot_pose, self._train_rng)
            # the new z axis is always pointing up
            next_state.set(spot, "x", new_spot_pose.x)
            next_state.set(spot, "y", new_spot_pose.y)
            next_state.set(spot, "z", new_spot_pose.z)
            next_state.set(spot, "qx", new_spot_pose.rot.x)
            next_state.set(spot, "qy", new_spot_pose.rot.y)
            next_state.set(spot, "qz", new_spot_pose.rot.z)
            next_state.set(spot, "qw", new_spot_pose.rot.w)
            # update the EE pose
            if 'Away' in action_name:
                # MoveAway always firs stow arm
                new_hand_pose_mat = self.spot_kinematics.compute_fk(new_spot_pose.to_matrix(), \
                                    DEFAULT_HAND_STOW_ANGLES)[-1]
                new_hand_pose = math_helpers.SE3Pose.from_matrix(new_hand_pose_mat)
                # new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            else:
                # Otherwise, assme hand is in fixed rel pose to body
                hand_pose = utils.get_se3_hand_pose_from_state(next_state, spot)
                # Assume the hand pose is relatively fixed to the spot pose
                new_hand_pose = new_spot_pose.mult(spot_pose.inverse().\
                                                mult(hand_pose))
                # new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
            # update stair pose if in hand
            stair_objs = state.get_objects(self._stairs_type)
            for stair_obj in stair_objs:
                if state.get(stair_obj, "held"):
                    # Assume the stair pose is relatively fixed to the spot pose
                    new_stair_pose = new_hand_pose.mult(DEFAULT_HAND2STAIR_TF)
                    next_state.set(stair_obj, "x", new_stair_pose.x)
                    next_state.set(stair_obj, "y", new_stair_pose.y)
                    next_state.set(stair_obj, "z", new_stair_pose.z)
                    next_state.set(stair_obj, "qx", new_stair_pose.rot.x)
                    next_state.set(stair_obj, "qy", new_stair_pose.rot.y)
                    next_state.set(stair_obj, "qz", new_stair_pose.rot.z)
                    next_state.set(stair_obj, "qw", new_stair_pose.rot.w)
                    break
            return next_state
        elif action_name == "MoveToOnStairsHandViewObject":
            # note that we actually didn't use the sampler in the simulation
            # assume the robot is stepping on the center of the stair
            spot, stair, tgt = action_objs[0], action_objs[1], action_objs[2]
            if not self._AppliedTo_holds(next_state, [stair, tgt]):
                return next_state
            if not self._Near_holds(next_state, [stair, tgt]):
                return next_state
            # first get the stair pose, which is the foot pose
            stair_pose = utils.get_se3_pose_from_state(next_state, stair)
            body_pose_on_stair = stair_pose.mult(DEFAULT_STAIR2BODY_ONSTAIR_TF)
            # body_pose_on_stair = add_noise_to_pose(body_pose_on_stair, self._train_rng)
            # the new z axis is always pointing up
            next_state.set(spot, "x", body_pose_on_stair.x)
            next_state.set(spot, "y", body_pose_on_stair.y)
            next_state.set(spot, "z", body_pose_on_stair.z)
            next_state.set(spot, "qx", body_pose_on_stair.rot.x)
            next_state.set(spot, "qy", body_pose_on_stair.rot.y)
            next_state.set(spot, "qz", body_pose_on_stair.rot.z)
            next_state.set(spot, "qw", body_pose_on_stair.rot.w)
            # update the EE pose
            new_hand_pose_mat = self.spot_kinematics.compute_fk(body_pose_on_stair.to_matrix(), \
                                DEFAULT_HAND_STOW_ANGLES)[-1]
            new_hand_pose = math_helpers.SE3Pose.from_matrix(new_hand_pose_mat)
            # new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
        elif action_name == "PickObjectFromTop":
            spot, stair = action_objs
            if not self._HandEmpty.holds(next_state, [spot]):
                return next_state
            if not self._Reachable.holds(next_state, [spot, stair]):
                return next_state
            # note that we actually didn't use the sampler in the simulation
            # assume the robot directly grasping the stair's top handle
            # and the stair is teleported to the spot's hand
            # First, move to hand to default holding pose
            body_pose = utils.get_se3_pose_from_state(next_state, spot)
            # body_pose = add_noise_to_pose(body_pose, self._train_rng)
            hand_pose_mat = self.spot_kinematics.compute_fk(body_pose.to_matrix(), \
                                    DEFAULT_HAND_HOLDING_ANGLES)[-1]
            new_hand_pose = math_helpers.SE3Pose.from_matrix(hand_pose_mat)
            # new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            hand2obj_tf = DEFAULT_HAND2STAIR_TF
            new_stair_pose = new_hand_pose.mult(hand2obj_tf)
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
            next_state.set(spot, "gripper_open_percentage", 0.0)
            next_state.set(stair, "x", new_stair_pose.x)
            next_state.set(stair, "y", new_stair_pose.y)
            next_state.set(stair, "z", new_stair_pose.z)
            next_state.set(stair, "qx", new_stair_pose.rot.x)
            next_state.set(stair, "qy", new_stair_pose.rot.y)
            next_state.set(stair, "qz", new_stair_pose.rot.z)
            next_state.set(stair, "qw", new_stair_pose.rot.w)
            # update the "held" attribute of stair
            next_state.set(stair, "held", 1.0)
        elif action_name == "PlaceObjectInFront":
            spot, stair, _ = action_objs
            if not self._Holding_holds(next_state, [spot, stair]):
                return next_state
            # note that we actually didn't use the sampler in the simulation
            # assume the robot directly placing the stair on the ground
            # and the stair is teleported to the default offset
            spot_pose = utils.get_se3_pose_from_state(next_state, spot)
            # This is a little bit noisy, stair not flattly on ground
            dump_hand_pose = spot_pose.mult(DEFAULT_DUMPED_TF)
            new_stair_pose_noisy = dump_hand_pose.mult(DEFAULT_HAND2STAIR_TF)
            new_stair_pose_yaw = math_helpers.Quat.from_matrix(new_stair_pose_noisy.to_matrix()).to_yaw()
            new_stair_pose = math_helpers.SE3Pose(x=new_stair_pose_noisy.x, y=new_stair_pose_noisy.y, 
                                                  z=new_stair_pose_noisy.z, \
                                                rot=math_helpers.Quat.from_yaw(new_stair_pose_yaw))
            # sample arm initial pose, body 2 hand
            # spot_pose = add_noise_to_pose(spot_pose, self._train_rng)
            hand_pose_mat = self.spot_kinematics.compute_fk(spot_pose.to_matrix(), \
                                            DEFAULT_HAND_STOW_ANGLES)[-1]
            hand_pose = math_helpers.SE3Pose.from_matrix(hand_pose_mat)
            # hand_pose = add_noise_to_pose(hand_pose, self._train_rng)
            stair_height = next_state.get(stair, "height")
            next_state.set(spot, "gripper_open_percentage", 1.0)
            next_state.set(spot, "ee_x", hand_pose.x)
            next_state.set(spot, "ee_y", hand_pose.y)
            next_state.set(spot, "ee_z", hand_pose.z)
            next_state.set(spot, "ee_qx", hand_pose.rot.x)
            next_state.set(spot, "ee_qy", hand_pose.rot.y)
            next_state.set(spot, "ee_qz", hand_pose.rot.z)
            next_state.set(spot, "ee_qw", hand_pose.rot.w)
            next_state.set(stair, "x", new_stair_pose.x)
            next_state.set(stair, "y", new_stair_pose.y)
            next_state.set(stair, "z", self.ground_z + stair_height)
            next_state.set(stair, "qx", new_stair_pose.rot.x)
            next_state.set(stair, "qy", new_stair_pose.rot.y)
            next_state.set(stair, "qz", new_stair_pose.rot.z)
            next_state.set(stair, "qw", new_stair_pose.rot.w)
            # update the "held" attribute of stair
            next_state.set(stair, "held", 0.0)
        elif action_name == "Calibrate":
            # first check if the spots hand sees the calibration target
            spot, obj = action_objs
            if not self._HandSees.holds(next_state, [spot, obj]):
                return next_state
            # then check if the tgt is the calibration tgt
            if not self._CalibrationTgt_holds(next_state, [spot, obj]):
                return next_state
            # finally, set the calibrated flag
            next_state.set(spot, "calibrated", 1)
            return next_state
        elif action_name == "Measure":
            # first check if the spots hand sees the target
            spot, obj = action_objs
            if not self._HandSees.holds(next_state, [spot, obj]):
                return next_state
            # then check if the spot is calibrated
            if not self._Calibrated_holds(next_state, [spot]):
                return next_state
            # finally, set the measured flag
            next_state.set(obj, "measured", 1)
            return next_state
        elif action_name == "HandViewObject":
            spot = action_objs[0]
            hand_rel_se3_pose = action_args[1] #body2newhand
            # update the EE pose
            body_pose = utils.get_se3_pose_from_state(state, spot)
            # Assume the hand pose is relatively fixed to the spot pose
            new_hand_pose = body_pose * hand_rel_se3_pose
            # new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
            return next_state
        elif action_name == "HackMoveBack":
            spot = action_objs[0]
            robot_rel_se2_pose = action_args[1]
            spot_pose = utils.get_se3_pose_from_state(next_state, spot)
            spot_se2_pose = spot_pose.get_closest_se2_transform()
            new_spot_se2_pose = spot_se2_pose * robot_rel_se2_pose
            new_spot_pose = new_spot_se2_pose.get_closest_se3_transform()
            # the new z of these move actions are always on ground
            new_spot_pose.z = self.spot_height + self.ground_z
            # new_spot_pose = add_noise_to_pose(new_spot_pose, self._train_rng)
            # the new z axis is always pointing up
            next_state.set(spot, "x", new_spot_pose.x)
            next_state.set(spot, "y", new_spot_pose.y)
            next_state.set(spot, "z", new_spot_pose.z)
            next_state.set(spot, "qx", new_spot_pose.rot.x)
            next_state.set(spot, "qy", new_spot_pose.rot.y)
            next_state.set(spot, "qz", new_spot_pose.rot.z)
            next_state.set(spot, "qw", new_spot_pose.rot.w)
            # update the EE pose
            new_hand_pose_mat = self.spot_kinematics.compute_fk(new_spot_pose.to_matrix(), \
                                DEFAULT_HAND_STOW_ANGLES)[-1]
            new_hand_pose = math_helpers.SE3Pose.from_matrix(new_hand_pose_mat)
            # new_hand_pose = add_noise_to_pose(new_hand_pose, self._train_rng)
            next_state.set(spot, "ee_x", new_hand_pose.x)
            next_state.set(spot, "ee_y", new_hand_pose.y)
            next_state.set(spot, "ee_z", new_hand_pose.z)
            next_state.set(spot, "ee_qx", new_hand_pose.rot.x)
            next_state.set(spot, "ee_qy", new_hand_pose.rot.y)
            next_state.set(spot, "ee_qz", new_hand_pose.rot.z)
            next_state.set(spot, "ee_qw", new_hand_pose.rot.w)
        else:
            raise ValueError(f"Unknown action name: {action_name}")
        return next_state
    
    def step(self, action: Action) -> Observation:
        """Override step() because simulate() is not implemented."""
        assert isinstance(action.extra_info, (list, tuple))
        action_name, action_objs, action_fn, action_fn_args = action.extra_info
        self._last_action = action
        # The extra info is (action name, objects, function, function args).
        # The action name is either an operator name (for use with nonpercept
        # predicates) or a special name. See below for the special names.

        obs = self._current_observation
        assert isinstance(obs, _SpotArmObservation)
        assert self.action_space.contains(action.arr)

        # Special case: the action is "done", indicating that the robot
        # believes it has finished the task. Used for goal checking.
        if action_name == "done":

            # During a dry run, trust that the goal is accomplished if the
            # done action is returned, since we don't want a human in the loop.
            if CFG.spot_run_dry:
                self._current_task_goal_reached = True
                return self._current_observation

            while True:
                goal_description = self._current_task.goal_description
                logging.info(f"The goal is: {goal_description}")
                prompt = "Is the goal accomplished? Answer y or n. "
                response = input(prompt).strip()
                if response == "y":
                    self._current_task_goal_reached = True
                    break
                if response == "n":
                    self._current_task_goal_reached = False
                    break
                logging.info("Invalid input, must be either 'y' or 'n'")
            return self._current_observation

        # Otherwise, the action is either an operator to execute or a special
        # action. The only difference between the two is that operators update
        # the non-perfect states.

        operator_names = {o.name for o in self._strips_operators}

        # The action corresponds to an operator finishing.
        # alsways set()
        next_nonpercept = obs.nonpercept_atoms

        if CFG.spot_run_dry:
            # Simulate the effect of the action using the environment's simulate
            state = self.obs2state(obs)
            next_state = self.simulate(state, action)
            next_obs = self.state2obs(next_state)

        else:
            # Execute the action in the real environment. Automatically retry
            # if a retryable error is encountered.
            while True:
                try:
                    action_fn(*action_fn_args)  # type: ignore
                    time.sleep(1.0)
                    break
                except RetryableRpcError as e:
                    logging.warning("WARNING: the following retryable error "
                                    f"was encountered. Trying again.\n{e}")

            # Get the new observation. Again, automatically retry if needed.
            while True:
                try:
                    state = self.obs2state(obs)
                    next_state_sim = self.simulate(state, action)
                    next_obs = self._build_observation(next_nonpercept, next_state_sim)
                    break
                except RetryableRpcError as e:
                    logging.warning("WARNING: the following retryable error "
                                    f"was encountered. Trying again.\n{e}")
        self._current_observation = next_obs
        return self._current_observation

    def get_observation(self) -> Observation:
        return self._current_observation

    def goal_reached(self) -> bool:
        return self._current_task_goal_reached

    def _build_observation(self,
                           ground_atoms: Set[GroundAtom],
                           state: State) -> _SpotArmObservation:
        """Helper for building a new _SpotArmObservation().

        This is an environment method because the nonpercept predicates
        may vary per environment.
        """
        # dummy_obs = _SpotArmObservation() # type: ignore
        # return dummy_obs
        # Make sure the robot pose is up to date.
        assert self._robot is not None
        assert self._localizer is not None
        self._localizer.localize()
        # Get the universe of all object detections.
        # all_object_detection_ids = set(self._detection_id_to_obj)
        # Get the camera images.
        time.sleep(0.5)
        # rgbds = capture_images(self._robot, self._localizer,
        #                        camera_names=["frontleft_fisheye_image", "frontright_fisheye_image"])
        # all_detections, all_artifacts = detect_objects(
        #     all_object_detection_ids, rgbds, self._allowed_regions)

        # if CFG.spot_render_perception_outputs:
        #     outdir = Path(CFG.spot_perception_outdir)
        #     time_str = time.strftime("%Y%m%d-%H%M%S")
        #     detections_outfile = outdir / f"detections_{time_str}.png"
        #     no_detections_outfile = outdir / f"no_detections_{time_str}.png"
        #     visualize_all_artifacts(all_artifacts, detections_outfile,
        #                             no_detections_outfile)

        # # Separately, get detections for the hand in particular.
        # hand_rgbd = {
        #     k: v
        #     for (k, v) in rgbds.items() if k == "hand_color_image"
        # }
        # hand_detections, hand_artifacts = detect_objects(
        #     all_object_detection_ids, hand_rgbd, self._allowed_regions)

        # if CFG.spot_render_perception_outputs:
        #     detections_outfile = outdir / f"hand_detections_{time_str}.png"
        #     no_detect_outfile = outdir / f"hand_no_detections_{time_str}.png"
        #     visualize_all_artifacts(hand_artifacts, detections_outfile,
        #                             no_detect_outfile)

        # # Also, get detections that every camera except the back camera can
        # # see. This is important for our 'InView' predicate.
        # non_back_camera_rgbds = {
        #     k: v
        #     for (k, v) in rgbds.items() if k in [
        #         "hand_color_image", "frontleft_fisheye_image",
        #         "frontright_fisheye_image"
        #     ]
        # }
        # non_back_detections, _ = detect_objects(all_object_detection_ids,
        #                                         non_back_camera_rgbds,
        #                                         self._allowed_regions)

        # # Now construct a dict of all objects in view, as well as a set
        # # of objects that the hand can see, and that all cameras except
        # # the back can see.
        # all_objects_in_view = {
        #     self._detection_id_to_obj[det_id]: val
        #     for (det_id, val) in all_detections.items()
        # }
        # self._last_known_object_poses.update(all_objects_in_view)
        # objects_in_hand_view = set(self._detection_id_to_obj[det_id]
        #                            for det_id in hand_detections)
        # objects_in_any_view_except_back = set(
        #     self._detection_id_to_obj[det_id]
        #     for det_id in non_back_detections)
        robot_pose = self._localizer.get_last_robot_pose()
        hand_pose = self._localizer.get_last_hand_pose()
        gripper_open_percentage = get_robot_gripper_open_percentage(
            self._robot) / 100

        robot_pose_sim = utils.get_se3_pose_from_state(state, self._spot_object)
        hand_pose_sim = utils.get_se3_hand_pose_from_state(state, self._spot_object)
        # check if they are aligned
        body_rot_y = robot_pose.rot.y
        body_error_trans = np.linalg.norm(np.array([robot_pose.x, robot_pose.y, robot_pose.z]) - \
                                            np.array([robot_pose_sim.x, robot_pose_sim.y, robot_pose_sim.z]))
        hand_error_trans = np.linalg.norm(np.array([hand_pose.x, hand_pose.y, hand_pose.z]) - \
                                            np.array([hand_pose_sim.x, hand_pose_sim.y, hand_pose_sim.z]))
        while (body_error_trans > 0.2 and (body_rot_y > -0.05)):
            logging.warning(f"On Ground Body pose error larger than expected")
            logging.info(body_rot_y)
            logging.info(f"Sim pose {robot_pose_sim}")
            logging.info(f"Real pose {robot_pose}")
            logging.info(f"Error {body_error_trans}")
            logging.info("Trying to hack move the robot")
            curr_body_se2 = robot_pose.get_closest_se2_transform()
            desired = robot_pose_sim.get_closest_se2_transform()
            input(f"Trying to adjust pose to {desired}, press enter to continue")
            navigate_to_absolute_pose(self._robot, self._localizer, desired)
            time.sleep(0.5)
            self._localizer.localize()
            # perception running in option now
            # rgbds = capture_images(self._robot, self._localizer,
            #                    camera_names=["frontleft_fisheye_image", "frontright_fisheye_image"])
            robot_pose = self._localizer.get_last_robot_pose()
            hand_pose = self._localizer.get_last_hand_pose()
            gripper_open_percentage = get_robot_gripper_open_percentage(
                self._robot) / 100
            body_error_trans = np.linalg.norm(np.array([robot_pose.x, robot_pose.y, robot_pose.z]) - \
                                            np.array([robot_pose_sim.x, robot_pose_sim.y, robot_pose_sim.z]))
            hand_error_trans = np.linalg.norm(np.array([hand_pose.x, hand_pose.y, hand_pose.z]) - \
                                            np.array([hand_pose_sim.x, hand_pose_sim.y, hand_pose_sim.z]))
        
        if hand_error_trans > 0.05:
            logging.warning(f"Hand pose error larger than expected")
            logging.info(f"Sim pose {hand_pose_sim}")
            logging.info(f"Real pose {hand_pose}")
            logging.info(f"Error {hand_error_trans}")
        if abs(gripper_open_percentage - state.get(self._spot_object, "gripper_open_percentage")) > 0.05:
            logging.warning(f"Gripper open percentage error larger than expected")
            logging.info(f"Actual open: {gripper_open_percentage}")

        sim_obs = self.state2obs(state)

        next_obs = _SpotArmObservation(
            images=None,
            objects_in_view=sim_obs.objects_in_view,
            objects_in_hand_view=set(),
            objects_in_any_view_except_back=set(),
            object_in_hand=sim_obs.object_in_hand,
            robot=self._spot_object,
            gripper_open_percentage=gripper_open_percentage,
            robot_pos=robot_pose,
            hand_pos=hand_pose,
            calibrated=sim_obs.calibrated,
            calibration_id=sim_obs.calibration_id,
            nonpercept_atoms=set(),
            nonpercept_predicates=set(),
        )

        return next_obs

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        # a dummy task gen function
        return self._get_tasks({},
                                num=0,
                                num_obj=2,
                                num_stairs=1,
                                rng=self._test_rng)
    
    def _generate_test_tasks(self, spot_home_pose) -> List[EnvironmentTask]:
        if CFG.in_domain_test:
            # use the same number of objects and stairs as training
            # no compositional generalization
            logging.info("Generating in-domain test tasks...")
            return self._get_tasks(spot_home_pose,
                                   num=CFG.num_test_tasks,
                                   num_obj=2,
                                   num_stairs=1,
                                   rng=self._test_rng)
        else:
            logging.info("Generating out-of-domain test tasks...")
            return self._get_tasks(spot_home_pose,
                                   num=CFG.num_test_tasks,
                                    num_obj=2,
                                    num_stairs=2,
                                    rng=self._test_rng)

    def _get_tasks(self, spot_home_pose, \
                   num, num_obj, num_stairs, rng) \
        -> List[EnvironmentTask]:
        tasks = []
        task_id = 0
        nav_convex_hulls = get_allowed_map_regions()
        stair_convex_hulls = get_allowed_stair_regions()
        obj_convex_hulls = get_allowed_obj_regions()
        while task_id < num:
            skip = False
            logging.info(f"Generating task {task_id}...")
            color_id = 0
            state_dict = {}
            # First, sample the spot robot
            # load spot_body initial pose
            spot_x = spot_home_pose["x"]
            spot_y = spot_home_pose["y"]
            angle = spot_home_pose["angle"]
            se2_pose = math_helpers.SE2Pose(x=spot_x, y=spot_y, angle=angle)
            body_pose = se2_pose.get_closest_se3_transform()
            body_pose.z = self.ground_z + self.spot_height
            spot_geom = utils.Circle(spot_x, spot_y, 1)
            spot_z = body_pose.z
            spot_qx = body_pose.rot.x
            spot_qy = body_pose.rot.y
            spot_qz = body_pose.rot.z
            spot_qw = body_pose.rot.w
            # sample arm initial pose, body 2 hand
            body_pose_mat = body_pose.to_matrix()
            hand_pose_mat = self.spot_kinematics.compute_fk(body_pose_mat, \
                                            DEFAULT_HAND_STOW_ANGLES)[-1]
            hand_pose = math_helpers.SE3Pose.from_matrix(hand_pose_mat)
            hand_x = hand_pose.x
            hand_y = hand_pose.y
            hand_z = hand_pose.z
            hand_qx = hand_pose.rot.x
            hand_qy = hand_pose.rot.y
            hand_qz = hand_pose.rot.z
            hand_qw = hand_pose.rot.w
            state_dict[self._spot_object] = {
                "gripper_open_percentage": 1.0, # initially always opened
                "x": spot_x, "y": spot_y, "z": spot_z,
                "qx": spot_qx, "qy": spot_qy, "qz": spot_qz, "qw": spot_qw,
                "ee_x": hand_x, "ee_y": hand_y, "ee_z": hand_z,
                "ee_qx": hand_qx, "ee_qy": hand_qy, "ee_qz": hand_qz, "ee_qw": hand_qw,
                "calibration_obj_id": 0,
                "calibrated": 0,
            }
            # we use a large circle to represent the spot robot so that
            # it initially is viewclear
            # sample tgt objects
            tgts = [Object(f"tgt{i}", self._tgt_obj_type) for i in range(num_obj)]
            stairs = [Object(f"stair{i}", self._stairs_type) for i in range(num_stairs)]
            # Sample initial poses for targets, making sure to keep
            # them far enough apart from one another.
            if num_stairs == 1:
                selected_hull = rng.choice(stair_convex_hulls)
                x1, y1 = sample_point_in_hull(selected_hull, rng)
                stair_height = 0.255
                suc = False
                while not suc:
                    suc, qx1, qy1, qz1, qw1 = self.sample_stair_using_arm(rng, \
                                    nav_convex_hulls, x1, y1, stair_height)
                r, g, b = self.all_colors[color_id]
                color_id += 1
                state_dict[stairs[0]] = {
                    "x": x1, "y": y1, "z": stair_height + self.ground_z,
                    "qx": qx1, "qy": qy1, "qz": qz1, "qw": qw1,
                    "shape": 1,
                    "height": stair_height,
                    "width": self.stair_sz,
                    "length": self.stair_sz,
                    "object_id": num_obj, # this is stair id
                    "held": 0.0, 
                    "lost": 0.0, # dry sim always false
                    "in_hand_view": 0.0, # dry sim always false
                    "in_view": 1.0, # dry sim always true
                    "r": r, "g": g, "b": b
                }
            else:
                assert num_stairs == 2
                while True:
                    selected_hull = rng.choice(stair_convex_hulls)
                    x1, y1 = sample_point_in_hull(selected_hull, rng)
                    x2, y2 = sample_point_in_hull(selected_hull, rng)
                    stair_height = 0.255
                    geom1 = utils.Circle(x1, y1, 0.9)
                    geom2 = utils.Circle(x2, y2, 0.9)
                    if not geom1.intersects(geom2):
                        suc1, qx1, qy1, qz1, qw1 = self.sample_stair_using_arm(rng, \
                                        nav_convex_hulls, x1, y1, stair_height)
                        suc2, qx2, qy2, qz2, qw2 = self.sample_stair_using_arm(rng, \
                                        nav_convex_hulls, x2, y2, stair_height)
                        if suc1 and suc2:
                            break
                pos = [(x1, y1), (x2, y2)]
                dis2spot = [np.linalg.norm(np.array([x1, y1]) - np.array([spot_x, spot_y])), \
                            np.linalg.norm(np.array([x2, y2]) - np.array([spot_x, spot_y]))]
                min_dis_idx = np.argmin(dis2spot)
                rot = [(qx1, qy1, qz1, qw1), (qx2, qy2, qz2, qw2)]
                for i, stair in enumerate(stairs):
                    r, g, b = self.all_colors[color_id]
                    color_id += 1
                    state_dict[stair] = {
                        "x": pos[i][0], "y": pos[i][1], "z": stair_height + self.ground_z,
                        "qx": rot[i][0], "qy": rot[i][1], "qz": rot[i][2], "qw": rot[i][3],
                        "shape": 1,
                        "height": stair_height,
                        "width": self.stair_sz,
                        "length": self.stair_sz,
                        "object_id": num_obj + i, # this is stair id
                        "held": 0.0, 
                        "lost": 0.0, # dry sim always false
                        "in_hand_view": 0.0, # dry sim always false
                        "in_view": 1.0, # dry sim always true
                        "r": r, "g": g, "b": b
                    }

            # Step 2: sample targets
            while True:
                selected_hull = rng.choice(obj_convex_hulls)
                x1, y1 = sample_point_in_hull(selected_hull, rng)
                x2, y2 = sample_point_in_hull(selected_hull, rng)
                geom1 = utils.Circle(x1, y1, 0.8)
                geom2 = utils.Circle(x2, y2, 0.8)
                if (not (geom1.intersects(geom2)) and \
                    not (geom1.intersects(spot_geom)) and \
                    not (geom2.intersects(spot_geom))):
                    # second object always need the stair
                    suc2, tgt_z2, qw2, qx2, qy2, qz2 = \
                        self.sample_tgt_using_arm(rng, nav_convex_hulls, tgt_x=x2, tgt_y=y2, \
                                        stair_height=stair_height, off_stair=False)
                    if num_stairs == 1:
                        # first object doesn't need the stair
                        suc1, tgt_z1, qw1, qx1, qy1, qz1 = \
                            self.sample_tgt_using_arm(rng, nav_convex_hulls, tgt_x=x1, tgt_y=y1, \
                                        stair_height=0, off_stair=True)
                    else:
                        suc1, tgt_z1, qw1, qx1, qy1, qz1 = \
                            self.sample_tgt_using_arm(rng, nav_convex_hulls, tgt_x=x1, tgt_y=y1, \
                                        stair_height=stair_height, off_stair=False)
                    if suc1 and suc2:
                        break
            pos = [(x1, y1, tgt_z1), (x2, y2, tgt_z2)]
            rot = [(qx1, qy1, qz1, qw1), (qx2, qy2, qz2, qw2)]
            if num_stairs == 1:
                stair_id = [-1, 2]
            else:
                # always make the nearest stair the first stair
                stair_id = [2, 3] if min_dis_idx == 0 else [3, 2]
            for i, tgt in enumerate(tgts):
                r, g, b = self.all_colors[color_id]
                color_id += 1
                tgt_id = int(tgt.name[3])
                state_dict[tgt] = {
                    "x": pos[i][0], "y": pos[i][1], "z": pos[i][2],
                    "qx": rot[i][0], "qy": rot[i][1], "qz": rot[i][2], "qw": rot[i][3],
                    "shape": 0,
                    "height": self.tgt_sz,
                    "width": self.tgt_sz,
                    "length": self.tgt_sz,
                    "object_id": tgt_id,
                    "measured": 0,
                    "stair_id": stair_id[i],
                    "measured": 0,
                    "lost": 0.0, # dry sim always false
                    "in_hand_view": 0.0, # dry sim always false
                    "in_view": 1.0, # dry sim always true
                    "r": r, "g": g, "b": b
                }

            if not skip:
                init_state = utils.create_state_from_dict(state_dict)
                goal = set()
                for tgt in tgts:
                    goal.add(GroundAtom(self._Measured, [tgt]))
                task = EnvironmentTask(init_state, goal)
                tasks.append(task)
                task_id += 1
                logging.info(f"Task {task_id} generated.")
            else:
                logging.warning(f"Task {task_id} skipped.")
        return tasks

    def get_test_tasks(self) -> List[EnvironmentTask]:
        """Return the ordered list of tasks for testing / evaluation."""
        if not self._test_tasks:
            tasks_fname, _ = utils.create_task_filename_str(train=False)
            if not os.path.exists(tasks_fname):
                logging.warning(f"File {tasks_fname} does not exist. "
                                "Generating and saving new ones.")
                spot_home_pose = load_spot_metadata()["spot-home-pose"]
                self._test_tasks = self._generate_test_tasks(spot_home_pose)
                os.makedirs(tasks_fname, exist_ok=True)
                os.makedirs(os.path.join(tasks_fname, "vis"), exist_ok=True)
                for idx, task in enumerate(self._test_tasks):
                    utils.task2json(task, os.path.join(tasks_fname, f'{idx}.json'))
                    fig_name = os.path.join(tasks_fname, "vis", f'{idx}.png')
                    self.render_state_plt(task.init, task, save_path=fig_name)
            else:
                assert os.path.exists(tasks_fname), f"File {tasks_fname} does not exist"
                files = natsorted(Path(tasks_fname).glob("*.json"))
                assert len(files) >= CFG.num_test_tasks
                test_tasks = [
                    self._load_task_from_json(f)
                    for f in files[:CFG.num_test_tasks]
                ]
                self._test_tasks = test_tasks
                # self.convert_tasks_to_spot(original_test_tasks,
                #                                             spot_home_pose)
        return self._test_tasks

    def _get_dry_task(self, train_or_test: str,
                      task_idx: int) -> EnvironmentTask:
        if train_or_test == "train":
            task = self._train_tasks[task_idx]
        else:
            assert train_or_test == "test", "Invalid train_or_test!"
            task = self._test_tasks[task_idx]
        init_obs = self.state2obs(task.init)
        # Finish the task.
        return EnvironmentTask(init_obs, task.goal)
    
    # hacking functions
    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, stair = objects
        #  get the x y z
        held = state.get(stair, "held")
        if held:
            return True
        return False
    
    def _SurroundingClear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        spot = objects[0]
        spot_pose = utils.get_se3_pose_from_state(state, spot)
        spot_x = spot_pose.x
        spot_y = spot_pose.y
        for tgt in state.get_objects(self._stairs_type):
            # for fast computation
            # can't use reachable, as if the spot is on stair, it is also not reachable
            # But in this case, it shoud not be surrounding clear
            # if self._Reachable.holds(state, [spot, tgt]):
            #     return False
            tgt_pose = utils.get_se3_pose_from_state(state, tgt)
            tgt_x = tgt_pose.x
            tgt_y = tgt_pose.y
            dist_2d = np.sqrt((spot_x - tgt_x)**2 + (spot_y - tgt_y)**2)
            # MoveAway definitely delete it
            if dist_2d < 0.8:
                return False
        return True
    
    def _HandEmpty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot = objects[0]
        for stair in state.get_objects(self._stairs_type):
            if self._Holding_holds(state=state, objects=[robot, stair]):
                return False
            
        return True

    def state2obs(self, state: State) -> _SpotArmObservation:
        # only used in dry sim
        # assert CFG.spot_run_dry, "Only used in dry sim!"
        # Create the objects and their initial poses.
        # Not just SE3Poses are needed

        # Assume all objects are in view all the time
        objects_in_view: Dict[Object, Dict] = {}
        object_in_hand = None

        # Assuming initially the robot views all objects
        for obj in state.get_objects(self._tgt_obj_type):
            objects_in_view[obj] = {}
            pose = utils.get_se3_pose_from_state(state, obj)
            objects_in_view[obj]["pose"] = pose
            objects_in_view[obj]["measured"] = state.get(obj, "measured")
            objects_in_view[obj]["stair_id"] = state.get(obj, "stair_id")
            objects_in_view[obj]["object_id"] = state.get(obj, "object_id")
            objects_in_view[obj]["rgb"] = (state.get(obj, "r"),
                                             state.get(obj, "g"),
                                             state.get(obj, "b"))

        for stair in state.get_objects(self._stairs_type):
            objects_in_view[stair] = {}
            pose = utils.get_se3_pose_from_state(state, stair)
            objects_in_view[stair]["pose"] = pose
            objects_in_view[stair]["object_id"] = state.get(stair, "object_id")
            objects_in_view[stair]["height"] = state.get(stair, "height")
            objects_in_view[stair]["rgb"] = (state.get(stair, "r"),
                                             state.get(stair, "g"),
                                             state.get(stair, "b"))
            if state.get(stair, "held"):
                object_in_hand = stair

        robot_obj = state.get_objects(self._spot_type)[0]
        assert robot_obj == self._spot_object, "Robot object mismatch!"
        robot_pose = utils.get_se3_pose_from_state(state, robot_obj)
        robot_hand_pose = utils.get_se3_hand_pose_from_state(state,
                                                             robot_obj)
        gripper_open_percentage = state.get(robot_obj, "gripper_open_percentage")
        spot_cal_id = state.get(robot_obj, "calibration_obj_id")
        calibrated = state.get(robot_obj, "calibrated")

        # Create the initial observation.
        obs = _SpotArmObservation(
            images={},
            objects_in_view=objects_in_view,
            objects_in_hand_view=set(),
            objects_in_any_view_except_back=set(),
            object_in_hand=object_in_hand,
            robot=self._spot_object,
            gripper_open_percentage=gripper_open_percentage,
            robot_pos=robot_pose,
            hand_pos=robot_hand_pose,
            calibrated=calibrated,
            calibration_id=spot_cal_id,
            nonpercept_atoms=set(),
            nonpercept_predicates=set(),
        )
        return obs
    
    def obs2state(self, obs: _SpotArmObservation) -> State:
        # only used in dry sim
        # assert CFG.spot_run_dry, "Only used in dry sim!"
        robot_pos = obs.robot_pos
        robot_hand_pos = obs.hand_pos
        gripper_open_percentage = obs.gripper_open_percentage
        calibrated = obs.calibrated
        calibration_id = obs.calibration_id
        state_dict = {
            self._spot_object: {
                "gripper_open_percentage": gripper_open_percentage,
                "x": robot_pos.x,
                "y": robot_pos.y,
                "z": robot_pos.z,
                "qw": robot_pos.rot.w,
                "qx": robot_pos.rot.x,
                "qy": robot_pos.rot.y,
                "qz": robot_pos.rot.z,
                "ee_x": robot_hand_pos.x,
                "ee_y": robot_hand_pos.y,
                "ee_z": robot_hand_pos.z,
                "ee_qw": robot_hand_pos.rot.w,
                "ee_qx": robot_hand_pos.rot.x,
                "ee_qy": robot_hand_pos.rot.y,
                "ee_qz": robot_hand_pos.rot.z,
                "calibrated": calibrated,
                "calibration_obj_id": calibration_id,
            },
        }
        for obj, info in obs.objects_in_view.items():
            object_id = int(info["object_id"])
            pose = info["pose"]
            rgb = info["rgb"]
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
            # dry sim always false
            state_dict[obj]["in_hand_view"] = 0.0
            state_dict[obj]["in_view"] = 1.0
            state_dict[obj]["lost"] = 0.0
            # Add static object features.
            # static_feats = self._static_object_features.get(obj.name, {})
            # state_dict[obj].update(static_feats)
            # Add initial features for movable objects.
            if obj.is_instance(_stair_object_type):
                # Detect if the object is in (held) currently.
                held_val = 1.0 if obj == obs.object_in_hand else 0.0
                state_dict[obj]["held"] = held_val
                # stair height
                state_dict[obj]["height"] = info["height"]
                # stair width and length
                state_dict[obj]["width"] = CFG.viewplan_stair_sz
                state_dict[obj]["length"] = CFG.viewplan_stair_sz
            elif obj.is_instance(_target_object_w_stair_type):
                # it can't be held
                # target width and length
                state_dict[obj]["width"] = self.tgt_sz
                state_dict[obj]["length"] = self.tgt_sz
                state_dict[obj]["height"] = self.tgt_sz
                state_dict[obj]["stair_id"] = info["stair_id"]
                state_dict[obj]["measured"] = info["measured"]
        # Construct a regular state before adding atoms.
        state = utils.create_state_from_dict(state_dict)
        return state