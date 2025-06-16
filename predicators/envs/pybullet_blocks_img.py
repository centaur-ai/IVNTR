"""A PyBullet version of Blocks."""

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple, \
    Optional, Sequence

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.blocks import BlocksEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Array, EnvironmentTask, Object, State, \
    Type, Predicate


class PyBulletBlocksEnv(PyBulletEnv, BlocksEnv):
    """PyBullet Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._block_type = Type("block", [
            "img",
            "pose_x", "pose_y", "pose_z", "held", "color_r", "color_g",
            "color_b"
        ])
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "pose_z", "fingers"])
        # Predicates
        self._On = Predicate("On", [self._block_type, self._block_type],
                             self._On_holds)
        self._OnTable = Predicate("OnTable", [self._block_type],
                                  self._OnTable_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._block_type],
                                  self._Holding_holds)
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        # Hyperparameters from CFG.
        self._block_size = CFG.blocks_block_size
        self._num_blocks_train = CFG.blocks_img_num_blocks_train
        self._num_blocks_test = CFG.blocks_img_num_blocks_test

        # We track the correspondence between PyBullet object IDs and Object
        # instances for blocks. This correspondence changes with the task.
        self._block_id_to_block: Dict[int, Object] = {}

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle blocks-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert using_gui, \
                "using_gui must be True to use pybullet_draw_debug."
            # Draw the workspace on the table for clarity.
            p.addUserDebugLine([cls.x_lb, cls.y_lb, cls.table_height],
                               [cls.x_ub, cls.y_lb, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_ub, cls.table_height],
                               [cls.x_ub, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_lb, cls.table_height],
                               [cls.x_lb, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_ub, cls.y_lb, cls.table_height],
                               [cls.x_ub, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            # Draw coordinate frame labels for reference.
            p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            # Draw the pick z location at the x/y midpoint.
            mid_x = (cls.x_ub + cls.x_lb) / 2
            mid_y = (cls.y_ub + cls.y_lb) / 2
            p.addUserDebugText("*", [mid_x, mid_y, cls.pick_z],
                               [1.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)

        # Create blocks. Note that we create the maximum number once, and then
        # later on, in reset_state(), we will remove blocks from the workspace
        # (teleporting them far away) based on which ones are in the state.
        num_blocks = max(max(CFG.blocks_img_num_blocks_train),
                         max(CFG.blocks_img_num_blocks_test))
        block_ids = []
        block_size = CFG.blocks_block_size
        for i in range(num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            half_extents = (block_size / 2.0, block_size / 2.0,
                            block_size / 2.0)
            block_ids.append(
                create_pybullet_block(color, half_extents, cls._obj_mass,
                                      cls._obj_friction, cls._default_orn,
                                      physics_client_id))
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._block_ids = pybullet_bodies["block_ids"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.robot_init_x, cls.robot_init_y, cls.robot_init_z),
                       robot_ee_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home)

    def _extract_robot_state(self, state: State) -> Array:
        # The orientation is fixed in this environment.
        qx, qy, qz, qw = self.get_robot_ee_home_orn()
        f = self.fingers_state_to_joint(self._pybullet_robot,
                                        state.get(self._robot, "fingers"))
        return np.array([
            state.get(self._robot, "pose_x"),
            state.get(self._robot, "pose_y"),
            state.get(self._robot, "pose_z"), qx, qy, qz, qw, f
        ],
                        dtype=np.float32)
    
    def _extract_block_centric_img(self, block: Object, pass_dummy: bool) -> Array:
        if pass_dummy:
            return np.zeros((CFG.pybullet_block_crop_size, CFG.pybullet_block_crop_size, 3), dtype=np.uint8)

        # Compute view matrix
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._camera_target,
            distance=self._camera_distance,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._physics_client_id)

        width = CFG.pybullet_camera_width
        height = CFG.pybullet_camera_height

        # Compute projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width / height),
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self._physics_client_id)

        # Capture camera image
        (_, _, px, _, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._physics_client_id)

        rgb_array = np.array(px).reshape((height, width, 4))[:, :, :3]

        # Get block 3D position
        for block_id, b in self._block_id_to_block.items():
            if b == block:
                (bx, by, bz), _ = p.getBasePositionAndOrientation(
                    block_id, physicsClientId=self._physics_client_id)
                break

        # Project to camera plane
        block_pose = np.array([bx, by, bz, 1.0])  # Homogeneous coordinates

        # Convert view and projection matrices to NumPy arrays
        view_matrix_np = np.array(view_matrix).reshape(4, 4).T  # Transpose to match NumPy format
        proj_matrix_np = np.array(proj_matrix).reshape(4, 4).T

        view_proj_matrix = proj_matrix_np @ view_matrix_np
        projected_point = view_proj_matrix @ block_pose

        # Normalize to screen coordinates
        projected_point /= projected_point[3]

        # Convert to pixel coordinates (flipping y-axis)
        px_x = int((projected_point[0] * 0.5 + 0.5) * width)
        px_y = int((1 - (projected_point[1] * 0.5 + 0.5)) * height)

        # Define crop size
        crop_size = CFG.pybullet_block_crop_size
        half_size = crop_size // 2

        # Ensure coordinates are within bounds
        px_x = np.clip(px_x, half_size, width - half_size)
        px_y = np.clip(px_y, half_size, height - half_size)

        # Crop block-centric image
        cropped_img = rgb_array[px_y - half_size:px_y + half_size, px_x - half_size:px_x + half_size]

        return cropped_img


    @classmethod
    def get_name(cls) -> str:
        return "pybullet_blocks_img"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle blocks-specific resetting."""
        super()._reset_state(state)

        # Reset blocks based on the state.
        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block = {}
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            self._block_id_to_block[block_id] = block_obj
            bx = state.get(block_obj, "pose_x")
            by = state.get(block_obj, "pose_y")
            bz = state.get(block_obj, "pose_z")
            p.resetBasePositionAndOrientation(
                block_id, [bx, by, bz],
                self._default_orn,
                physicsClientId=self._physics_client_id)
            # Update the block color. RGB values are between 0 and 1.
            r = state.get(block_obj, "color_r")
            g = state.get(block_obj, "color_g")
            b = state.get(block_obj, "color_b")
            color = (r, g, b, 1.0)  # alpha = 1.0
            p.changeVisualShape(block_id,
                                linkIndex=-1,
                                rgbaColor=color,
                                physicsClientId=self._physics_client_id)

        # Check if we're holding some block.
        held_block = self._get_held_block(state)
        if held_block is not None:
            self._force_grasp_object(held_block)

        # For any blocks not involved, put them out of view.
        h = self._block_size
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            assert block_id not in self._block_id_to_block
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, i * h],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        # if not reconstructed_state.allclose(state):
        #     logging.debug("Desired state:")
        #     logging.debug(state.pretty_str())
        #     logging.debug("Reconstructed state:")
        #     logging.debug(reconstructed_state.pretty_str())
        #     raise ValueError("Could not reconstruct state.")

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._block_id_to_block and self._held_obj_id. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        state_dict = {}

        # Get robot state.
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(rf)
        state_dict[self._robot] = np.array([rx, ry, rz, fingers])
        joint_positions = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            held = (block_id == self._held_obj_id)
            visual_data = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0]
            img_data = self._extract_block_centric_img(block, pass_dummy=CFG.blocks_img_dry_run)
            r, g, b, _ = visual_data[7]
            # pose_x, pose_y, pose_z, held
            state_dict[block] = np.array([img_data, bx, by, bz, held, r, g, b])

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num_tasks, possible_num_blocks, rng)
        return self._add_pybullet_state_to_tasks(tasks)
    
    def _sample_state_from_piles(self, piles: List[List[Object]],
                                 rng: np.random.Generator) -> State:
        state_dict: Dict[Object, Array] = {}
        # Create objects
        block_to_pile_idx = {}
        for i, pile in enumerate(piles):
            for j, block in enumerate(pile):
                assert block not in block_to_pile_idx
                block_to_pile_idx[block] = (i, j)
        # Sample pile (x, y)s
        pile_to_xy: Dict[int, Tuple[float, float]] = {}
        for i in range(len(piles)):
            pile_to_xy[i] = self._sample_initial_pile_xy(
                rng, set(pile_to_xy.values()))
        # Create block states
        color_idx = 0
        for block, pile_idx in block_to_pile_idx.items():
            pile_i, pile_j = pile_idx
            x, y = pile_to_xy[pile_i]
            z = self.table_height + self._block_size * (0.5 + pile_j)
            r, g, b, _ = self._obj_colors[color_idx % len(self._obj_colors)]
            # create dummy image first, real image will be added once the state is reset
            img = self._extract_block_centric_img(block, pass_dummy=True)
            # [img, pose_x, pose_y, pose_z, held, color_r, color_g, color_b]
            state_dict[block] = np.array([img, x, y, z, 0.0, r, g, b])
            color_idx += 1
        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = 1.0  # fingers start out open
        state_dict[self._robot] = [rx, ry, rz, rf]
        return State(state_dict)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        if CFG.pybullet_robot == "panda":
            # gripper rotated 90deg so parallel to x-axis
            normal = np.array([1., 0., 0.], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch":
            # gripper parallel to y-axis
            normal = np.array([0., 1., 0.], dtype=np.float32)
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        return {
            self._pybullet_robot.left_finger_id: normal,
            self._pybullet_robot.right_finger_id: -1 * normal,
        }

    def _force_grasp_object(self, block: Object) -> None:
        block_to_block_id = {b: i for i, b in self._block_id_to_block.items()}
        block_id = block_to_block_id[block]
        # The block should already be held. Otherwise, the position of the
        # block was wrong in the state.
        held_obj_id = self._detect_held_object()
        assert block_id == held_obj_id
        # Create the grasp constraint.
        self._held_obj_id = block_id
        self._create_grasp_constraint()

    @classmethod
    def fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
                               fingers_state: float) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either pybullet_robot.closed_fingers or
        pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = pybullet_robot.open_fingers
        closed_f = pybullet_robot.closed_fingers
        return closed_f if fingers_state == 0.0 else open_f

    def _fingers_joint_to_state(self, fingers_joint: float) -> float:
        """Convert the finger joint values in PyBullet to values for the State.

        The joint values given as input are the ones coming out of
        self._pybullet_robot.get_state().
        """
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
        # Fingers in the State should be either 0 or 1.
        return int(fingers_joint > (open_f + closed_f) / 2)

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        if state.get(block1, "held") >= self.held_tol or \
           state.get(block2, "held") >= self.held_tol:
            return False
        x1 = state.get(block1, "pose_x")
        y1 = state.get(block1, "pose_y")
        z1 = state.get(block1, "pose_z")
        x2 = state.get(block2, "pose_x")
        y2 = state.get(block2, "pose_y")
        z2 = state.get(block2, "pose_z")
        return np.allclose([x1, y1, z1], [x2, y2, z2 + self._block_size],
                           atol=self.on_tol)

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        z = state.get(block, "pose_z")
        desired_z = self.table_height + self._block_size * 0.5
        return (state.get(block, "held") < self.held_tol) and \
            (desired_z-self.on_tol < z < desired_z+self.on_tol)

    @staticmethod
    def _GripperOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        rf = state.get(robot, "fingers")
        # assert rf in (0.0, 1.0)
        return rf >= 0.5

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return self._get_held_block(state) == block

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        if self._Holding_holds(state, objects):
            return False
        block, = objects
        for other_block in state:
            if other_block.type != self._block_type:
                continue
            if self._On_holds(state, [other_block, block]):
                return False
        return True

    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            if state.get(block, "held") >= self.held_tol:
                return block
        return None

    def _get_block_at_xyz(self, state: State, x: float, y: float,
                          z: float) -> Optional[Object]:
        close_blocks = []
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            if np.allclose([x, y, z], block_pose, atol=self.pick_tol):
                dist = np.linalg.norm(np.array([x, y, z]) - block_pose)
                close_blocks.append((block, float(dist)))
        if not close_blocks:
            return None
        return min(close_blocks, key=lambda x: x[1])[0]  # min distance

    def _get_highest_block_below(self, state: State, x: float, y: float,
                                 z: float) -> Optional[Object]:
        blocks_here = []
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            block_pose = np.array(
                [state.get(block, "pose_x"),
                 state.get(block, "pose_y")])
            block_z = state.get(block, "pose_z")
            if np.allclose([x, y], block_pose, atol=self.pick_tol) and \
               block_z < z - self.pick_tol:
                blocks_here.append((block, block_z))
        if not blocks_here:
            return None
        return max(blocks_here, key=lambda x: x[1])[0]  # highest z