"""Blocks-On domain.

This environment IS downward refinable and DOESN'T require any
backtracking (as long as all the blocks can fit comfortably on the
table, which is true here because the block size and number of blocks
are much less than the table dimensions).
This environment has "GoalAchieved" as a terminal predicate and "AchieveGoal"
as the final option. This requires the PI algorithm to invent all the Predicates, 
including "On", "Clear", and "Hold".
This environment also does not have "held" as low-level feature, requiring the PI algorithm
to invent the "Holding" BINARY predicate.
On and Clear are preconditions of the AchieveGoal option.
"""

from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from bosdyn.client import math_helpers
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
from typing import Sequence, Set, List, Dict, Tuple, Optional

from predicators.envs.blocks_onclear import BlocksOnClearEnv
import numpy as np
import json
from pytorch3d.structures import Meshes
from predicators import utils
from gym.spaces import Box
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Type, Action, Array, \
    GroundAtom, EnvironmentTask

import open3d as o3d
import torch
from open3d.visualization import rendering
from scipy.spatial.distance import cdist


def generate_top_face_multi_gaussian(
    n=20,
    bumps=None,
    cube_size=1.0
):
    """
    Generate the top face of a cube with multiple 2D Gaussian bumps or depressions.
    :param n:        Grid resolution in x and y for the top face.
    :param bumps:    A list of dicts, each with keys:
                       { 'A': float, 'x0': float, 'y0': float, 'sigma': float }
                     representing a single Gaussian bump/depression.
    :param cube_size: The width/height/depth of the overall cube.
    :return: (vertices, triangles) as NumPy arrays
    """
    if bumps is None:
        bumps = []

    # x, y in [0, cube_size]
    xs = np.linspace(0, cube_size, n)
    ys = np.linspace(0, cube_size, n)

    vertices = []
    for y in ys:
        for x in xs:
            # Start z at the top of the cube
            z = cube_size
            # Add contribution from each bump
            for bump in bumps:
                A = bump["A"]
                x0 = bump["x0"]
                y0 = bump["y0"]
                sigma = bump["sigma"]
                z += A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            vertices.append([x, y, z])
    vertices = np.array(vertices, dtype=np.float32)

    # Create triangles (two per grid cell)
    triangles = []
    for j in range(n - 1):
        for i in range(n - 1):
            idx0 = j * n + i
            idx1 = j * n + (i + 1)
            idx2 = (j + 1) * n + i
            idx3 = (j + 1) * n + (i + 1)
            triangles.append([idx0, idx1, idx2])
            triangles.append([idx2, idx1, idx3])
    triangles = np.array(triangles, dtype=np.int32)

    return vertices, triangles


def create_multi_bump_cube_components_pytorch3d(n=20, bumps=None, cube_size=1.0):
    """
    Create two Meshes objects:
      1) The cube minus top (base mesh)
      2) The top face with multiple Gaussian bumps
    """
    # Step A: Parametric top
    top_vertices_np, top_triangles_np = generate_top_face_multi_gaussian(
        n=n, bumps=bumps, cube_size=cube_size
    )
    top_verts_torch = torch.from_numpy(top_vertices_np)   # (V_top, 3)
    top_faces_torch = torch.from_numpy(top_triangles_np)  # (F_top, 3)
    top_mesh = Meshes(verts=[top_verts_torch], faces=[top_faces_torch])

    # Step B: Cube minus top
    base_cube_verts = torch.tensor([
        [0,         0,          0],
        [cube_size, 0,          0],
        [cube_size, cube_size,  0],
        [0,         cube_size,  0],
        [0,         0,          cube_size],
        [cube_size, 0,          cube_size],
        [cube_size, cube_size,  cube_size],
        [0,         cube_size,  cube_size]
    ], dtype=torch.float32)

    base_cube_faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],       # bottom
        [4, 5, 6], [4, 6, 7],       # top (to remove)
        [0, 1, 5], [0, 5, 4],       # front
        [2, 3, 7], [2, 7, 6],       # back
        [0, 3, 7], [0, 7, 4],       # left
        [1, 2, 6], [1, 6, 5],       # right
    ], dtype=torch.int64)

    top_faces = {(4, 5, 6), (4, 6, 7)}
    filtered_faces = []
    for f in base_cube_faces:
        f_sorted = tuple(sorted(f.tolist()))
        if f_sorted not in top_faces:
            filtered_faces.append(f)
    filtered_faces = torch.stack(filtered_faces)

    cube_minus_top_mesh = Meshes(verts=[base_cube_verts], faces=[filtered_faces])

    return cube_minus_top_mesh, top_mesh

# Custom handler for square patches with markers
class CustomSquareHandler(HandlerPatch):
    def __init__(self, marker=None, marker_position=(0.5, 0.5), **kwargs):
        super().__init__(**kwargs)
        self.marker = marker
        self.marker_position = marker_position

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        square = mpatches.Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            transform=trans
        )
        artists = [square]
        if self.marker:
            marker_x = xdescent + self.marker_position[0] * width
            marker_y = ydescent + self.marker_position[1] * height
            marker_artist = plt.Line2D(
                [marker_x], [marker_y],
                marker=self.marker,
                color='black',
                markersize=fontsize * 0.8
            )
            artists.append(marker_artist)
        return artists

class BlocksEngraveVecEnv(BlocksOnClearEnv):
    """BlocksOnClear domain."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # parameters re-written from the BlocksEnv class
        self._block_size = CFG.blocks_engrave_block_size
        self.blo_x_lb = 0.0 + CFG.blocks_engrave_sigma[0] * 3
        self.blo_x_ub = self._block_size - CFG.blocks_engrave_sigma[0] * 3
        self.blo_y_lb = 0.0 + CFG.blocks_engrave_sigma[0] * 3
        self.blo_y_ub = self._block_size - CFG.blocks_engrave_sigma[0] * 3
        self.engrave_height = CFG.blocks_engrave_height
        self.table_height = 0.0
        # The table x bounds are (1.1, 1.6), but the workspace is smaller.
        # Make it narrow enough that blocks can be only horizontally arranged.
        # Note that these boundaries are for the block positions, and that a
        # block's origin is its center, so the block itself may extend beyond
        # the boundaries while the origin remains in bounds.
        self.x_lb = 0.0
        self.x_ub = 1.0
        # The table y bounds are (0.3, 1.2), but the workspace is smaller.
        self.y_lb = 0.0
        self.y_ub = 1.0
        self.pick_z = 1.0
        self.robot_init_x = (self.x_lb + self.x_ub) / 2
        self.robot_init_y = (self.y_lb + self.y_ub) / 2
        self.robot_init_z = self.pick_z
        self._block_type = Type("block", [
            "pose_x", "pose_y", "pose_z", 
            "irr_x", "irr_y", "irr_h",
            "color_r", "color_g",
            "color_b", "faceup", "goal_achieved"
        ]) # we don't need "held" in low-level features, should be predicates.
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "pose_z", "fingers"])
        # Predicates
        self._On = Predicate("On", [self._block_type, self._block_type],
                             self._On_holds)
        self._Matched = Predicate("Matched", [self._block_type, self._block_type],
                                    self._Matched_holds)
        self._Single = Predicate("Single", [self._block_type], self._Single_holds)
        self._FaceUp = Predicate("FaceUp", [self._block_type], self._FaceUp_holds)
        self._FaceDown = Predicate("FaceDown", [self._block_type], self._FaceDown_holds)
        self._OnTable = Predicate("OnTable", [self._block_type],
                                  self._OnTable_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._robot_type, self._block_type],
                                  self._Holding_holds) # Binary
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)

        self._GoalAchieved = Predicate("GoalAchieved", 
                [self._block_type, self._block_type], self._GoalAchieved_holds)
        self._NotEq = Predicate("NotEq", [self._block_type, self._block_type],
                            self._NotEq_holds)
        
        self._robot = Object("robby", self._robot_type)
        self.all_colors = [
            [0.0, 0.0, 1.0],  # blue
            [0.0, 1.0, 0.0],  # green
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
        self._num_blocks_train = CFG.blocks_engrave_num_blocks_train
        self._num_blocks_test = CFG.blocks_engrave_num_blocks_test

    @classmethod
    def get_name(cls) -> str:
        return "blocks_engrave_vec"

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, fingers, rotate, eng_x, eng_y, eng_h, achieve_goal]
        lowers = np.array([self.x_lb, self.y_lb, 0.0, 0.0, \
                           0.0, self.blo_x_lb, self.blo_y_lb, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 2.0, 1.0, 
                           1.0, self.blo_x_ub, self.blo_y_ub, self.engrave_height, 1.0], dtype=np.float32)
        return Box(lowers, uppers)
    
    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._GoalAchieved}
    
    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnTable, self._GripperOpen, self._Holding,
            self._Clear, self._GoalAchieved, self._Matched, self._FaceUp, 
            self._FaceDown, self._Single, self._NotEq
        }
    
    def _sample_state_from_piles(self, piles: List[List[Object]],
                                 rng: np.random.Generator) -> State:
        state_dict = {}
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
            r, g, b = self.all_colors[color_idx]
            # [pose_x, pose_y, pose_z, color_r, color_g, color_b, goal_achieved]
            face_up = 1.0 if pile_j == 0 else rng.choice([0.0, 1.0])
            eng_x = rng.uniform(self.blo_x_lb, self.blo_x_ub)
            eng_y = rng.uniform(self.blo_y_lb, self.blo_y_ub)
            eng_h = self.engrave_height
            state_dict[block] = {
                "pose_x": x,
                "pose_y": y,
                "pose_z": z,
                "color_r": r,
                "color_g": g,
                "color_b": b,
                "faceup": face_up,
                "goal_achieved": 0.0,
                "irr_x": eng_x,
                "irr_y": eng_y,
                "irr_h": eng_h
            }
            color_idx += 1
        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = 1.0  # fingers start out open
        state_dict[self._robot] = {
            "pose_x": rx,
            "pose_y": ry,
            "pose_z": rz,
            "fingers": rf
        }
        return utils.create_state_from_dict(state_dict)
    
    def _sample_initial_piles(self, num_blocks: int,
                              rng: np.random.Generator) -> List[List[Object]]:
        piles: List[List[Object]] = []
        for block_num in range(num_blocks):
            block = Object(f"block{block_num}", self._block_type)
            # If coin flip, start new pile
            if (block_num == 0) or (rng.uniform() < 0.2) and len(piles) < 2:
                piles.append([])
            # Add block to pile
            piles[-1].append(block)
        return piles
    
    def _sample_goal_from_piles(self, num_blocks: int,
                                piles: List[List[Object]],
                                rng: np.random.Generator,
                                init_state: State) -> Tuple[Set[GroundAtom], State]:
        # Sample goal pile that is different from initial
        new_state = init_state.copy()
        while True:
            goal_piles: List[List[Object]] = [[]]
            for block_num in rng.permutation(range(num_blocks)):
                block = Object(f"block{block_num}", self._block_type)
                if len(goal_piles[-1]) == 2:
                    goal_piles.append([])
                # Add block to pile
                goal_piles[-1].append(block)
            if goal_piles != piles:
                break
        # Create goal from piles
        goal_atoms = set()
        for pile in goal_piles:
            if len(pile) > 1:
                goal_atoms.add(GroundAtom(self._GoalAchieved, [pile[-1], pile[-2]]))
                # if rng.random() < 0.5:
                #     # initially they are matched
                #     eng_x = rng.uniform(self.blo_x_lb, self.blo_x_ub)
                #     eng_y = rng.uniform(self.blo_y_lb, self.blo_y_ub)
                #     eng_h = self.engrave_height
                #     mirrored_x = self.blo_x_ub - eng_x + self.blo_x_lb
                #     new_state.set(pile[-1], "irr_x", eng_x)
                #     new_state.set(pile[-1], "irr_y", eng_y)
                #     new_state.set(pile[-1], "irr_h", eng_h)
                #     new_state.set(pile[-2], "irr_x", mirrored_x)
                #     new_state.set(pile[-2], "irr_y", eng_y)
                #     new_state.set(pile[-2], "irr_h", -eng_h)
                    # assert self._Matched_holds(new_state, [pile[-1], pile[-2]])
        return goal_atoms, new_state
    
    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks)
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            while True:  # repeat until goal is not satisfied
                # modify the init state by creating random single and matched blocks
                goal, new_state = self._sample_goal_from_piles(num_blocks, piles, rng, init_state)
                if not all(goal_atom.holds(new_state) for goal_atom in goal):
                    break
            tasks.append(EnvironmentTask(new_state, goal))
        return tasks
    
    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        x, y, z, fingers, rotate,\
        eng_x, eng_y, eng_h, achieve_goal = action.arr
        # Infer which transition function to follow
        if achieve_goal == 1:
            return self._transition_achievegoal(state, x, y, z)
        if rotate != 0:
            return self._transition_rotate(state, x, y, z, rotate)
        if eng_h > 0.0:
            return self._transition_engrave(state, 
                                            x, y, z,
                                            eng_x, eng_y, eng_h)
        if fingers < 0.5:
            return self._transition_pick(state, x, y, z)
        if z < self.table_height + self._block_size:
            return self._transition_putontable(state, x, y, z)
        return self._transition_stack(state, x, y, z)
    
    def _transition_engrave(self, state: State, x: float, y: float, z: float,
                            eng_x: float, eng_y: float, eng_h: float) -> State:
        next_state = state.copy()
        top_block = self._get_block_at_xyz(state, x, y, z)
        if top_block is None:  # no block at this pose
            return next_state
        if not self._block_is_clear(top_block, state):
            return next_state
        if not self._Single_holds(state, [top_block]):
            return next_state
        bottom_block = self._get_highest_block_below(state, x, y, z)
        if bottom_block is None:
            return next_state
        if not self._OnTable_holds(state, [bottom_block]):
            return next_state
        if not self._On_holds(state, [top_block, bottom_block]):
            return next_state
        if not self._Single_holds(state, [bottom_block]):
            return next_state
        next_state.set(top_block, "irr_x", eng_x)
        next_state.set(top_block, "irr_y", eng_y)
        next_state.set(top_block, "irr_h", eng_h)
        next_state.set(bottom_block, "irr_x", eng_x)
        # need to mirror the x coordinates
        next_state.set(bottom_block, "irr_y", eng_y)
        next_state.set(bottom_block, "irr_h", -eng_h)
        mirrored_x = self.blo_x_ub - eng_x + self.blo_x_lb
        next_state.set(bottom_block, "irr_x", mirrored_x)
        return next_state
    
    def _transition_rotate(self, state: State, x: float, y: float, z: float,
                            rotate: float) -> State:
        next_state = state.copy()
        block = self._get_block_at_xyz(state, x, y, z)
        if block is None:  # no block at this pose
            return next_state
        if not self._Holding_holds(state, [self._robot, block]):
            return next_state
        assert rotate > 0
        current_faceup = state.get(block, "faceup")
        next_state.set(block, "faceup", 1.0 - current_faceup)
        return next_state
        
    def _transition_achievegoal(self, state: State, x: float, y: float, z: float) -> State:
        next_state = state.copy()
        top_block = self._get_block_at_xyz(state, x, y, z)
        if top_block is None:  # no block at this pose
            return next_state
        if not self._block_is_clear(top_block, state):
            return next_state
        if not self._FaceDown_holds(state, [top_block]):
            return next_state
        
        bottom_block = self._get_highest_block_below(state, x, y, z)
        if bottom_block is None:
            return next_state
        if not self._On_holds(state, [top_block, bottom_block]):
            return next_state
        if not self._Matched_holds(state, [top_block, bottom_block]):
            return next_state
        if not self._FaceUp_holds(state, [bottom_block]):
            return next_state
        next_state.set(top_block, "goal_achieved", 0.5)
        next_state.set(bottom_block, "goal_achieved", 1.0)
        return next_state
    
    def _transition_pick(self, state: State, x: float, y: float,
                         z: float) -> State:
        next_state = state.copy()
        # Can only pick if fingers are open
        if not self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_block_at_xyz(state, x, y, z)
        if block is None:  # no block at this pose
            return next_state
        # Can only pick if object is clear
        if not self._block_is_clear(block, state):
            return next_state
        # Execute pick
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        # Pick up block to a height of half the block size, to delete "On" in low level
        next_state.set(block, "pose_z", self.pick_z)
        next_state.set(self._robot, "fingers", 0.0)  # close fingers
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        next_state.set(self._robot, "pose_z", self.pick_z)
        return next_state
    
    def _transition_putontable(self, state: State, x: float, y: float,
                               z: float) -> State:
        next_state = state.copy()
        # Can only putontable if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_held_block(state)
        if block is None:
            return next_state
        # Check that table surface is clear at this pose
        poses = [[
            state.get(b, "pose_x"),
            state.get(b, "pose_y"),
            state.get(b, "pose_z")
        ] for b in state if b.is_instance(self._block_type)]
        existing_xys = {(float(p[0]), float(p[1])) for p in poses}
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        # Execute putontable
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        next_state.set(self._robot, "pose_z", z)
        return next_state
    
    def _transition_stack(self, state: State, x: float, y: float,
                          z: float) -> State:
        next_state = state.copy()
        # Can only stack if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        # Check that both blocks exist
        block = self._get_held_block(state)
        if block is None:
            return next_state
        other_block = self._get_highest_block_below(state, x, y, z)
        if other_block is None:  # no block to stack onto
            return next_state
        # Can't stack onto yourself!
        if block == other_block:
            return next_state
        # Need block we're stacking onto to be clear
        if not self._block_is_clear(other_block, state):
            return next_state
        # Execute stack by snapping into place
        cur_x = state.get(other_block, "pose_x")
        cur_y = state.get(other_block, "pose_y")
        cur_z = state.get(other_block, "pose_z")
        next_state.set(block, "pose_x", cur_x)
        next_state.set(block, "pose_y", cur_y)
        next_state.set(block, "pose_z", cur_z + self._block_size)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers
        next_state.set(self._robot, "pose_x", cur_x)
        next_state.set(self._robot, "pose_y", cur_y)
        next_state.set(self._robot, "pose_z", cur_z + self._block_size)
        return next_state
    
    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
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
        return (desired_z-self.on_tol < z < desired_z+self.on_tol)
    
    def _GoalAchieved_holds(self, state: State, objects: Sequence[Object]) -> bool:
        top_block, bottom_block = objects
        goal_achieve1 = state.get(top_block, "goal_achieved")
        goal_achieve2 = state.get(bottom_block, "goal_achieved")
        order = goal_achieve1 < goal_achieve2
        achieved = goal_achieve1 > 0
        if order and achieved:
            return True
        return False
    
    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        for other_block in state:
            if other_block.type != self._block_type:
                continue
            if self._On_holds(state, [other_block, block]):
                return False
        return True
    
    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, block = objects
        x_r = state.get(robot, "pose_x")
        y_r = state.get(robot, "pose_y")
        z_r = state.get(robot, "pose_z")
        f_r = state.get(robot, "fingers")

        x_b = state.get(block, "pose_x")
        y_b = state.get(block, "pose_y")
        z_b = state.get(block, "pose_z")
        if f_r > 0.5:
            # gripper is open
            return False
        else:
            # gripper is closed
            return np.isclose([x_r, y_r, z_r], [x_b, y_b, z_b]).all()
        
    def _FaceUp_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "faceup") > 0.5
    
    def _FaceDown_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "faceup") < 0.5
    
    def _Matched_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        x_1 = state.get(block1, "irr_x")
        y_1 = state.get(block1, "irr_y")
        h_1 = state.get(block1, "irr_h")
        x_2 = state.get(block2, "irr_x")
        y_2 = state.get(block2, "irr_y")
        h_2 = state.get(block2, "irr_h")
        x_ok = np.isclose(x_1 + x_2, self.blo_x_ub + self.blo_x_lb)
        y_ok = np.isclose(y_1, y_2)
        h_ok = np.isclose(h_1, -h_2)
        return x_ok and y_ok and h_ok
        
    def _Single_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        # the block is unique in this world, no match yet
        for other_block in state.get_objects(self._block_type):
            if self._Matched_holds(state, [block, other_block]):
                return False
        return True

    def _NotEq_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        block1_x = state.get(block1, "pose_x")
        block1_y = state.get(block1, "pose_y")
        block1_z = state.get(block1, "pose_z")
        block2_x = state.get(block2, "pose_x")
        block2_y = state.get(block2, "pose_y")
        block2_z = state.get(block2, "pose_z")
        dist3d = np.linalg.norm([block1_x-block2_x, block1_y-block2_y, block1_z-block2_z])
        return dist3d > self._block_size // 2

    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            if self._Holding_holds(state, [self._robot, block]):
                return block
            
    def render_state_plt(
        self,
        state,
        task,
        action: Optional[object] = None,
        save_path: Optional[str] = None,
        caption: Optional[str] = None
    ) -> plt.Figure:
        """Render state with two subplots for x-z, y-z, and a third x-y subplot for block display."""

        # --- 1) Create the figure and GridSpec for two rows, two columns ---
        #     Top row:    xz_ax (0,0), yz_ax (0,1)
        #     Bottom row: block_display_ax spans both columns (1,:)
        fig = plt.figure(figsize=(20, 14))
        gs = plt.GridSpec(
            2, 2,
            height_ratios=[6, 4],  # More space for the main row, less for the bottom
            wspace=0.3,
            hspace=0.3
        )

        # --- 2) Create the top-row subplots for x-z and y-z ---
        xz_ax = fig.add_subplot(gs[0, 0])
        yz_ax = fig.add_subplot(gs[0, 1])

        # Set up x-z axis
        r = self._block_size * 0.5
        xz_ax.set_xlabel("x", fontsize=18)
        xz_ax.set_ylabel("z", fontsize=18)
        xz_ax.set_xlim(self.x_lb - 2*r, self.x_ub + 2*r)
        xz_ax.set_ylim(self.table_height, self.table_height + r*20 + 0.1)

        # Set up y-z axis
        yz_ax.set_xlabel("y", fontsize=18)
        yz_ax.set_ylabel("z", fontsize=18)
        yz_ax.set_xlim(self.y_lb - 2*r, self.y_ub + 2*r)
        yz_ax.set_ylim(self.table_height, self.table_height + r*20 + 0.1)

        # --- 3) Collect and draw blocks on top subplots ---
        blocks = [o for o in state if o.is_instance(self._block_type)]

        for block in sorted(blocks):
            x = state.get(block, "pose_x")
            y = state.get(block, "pose_y")
            z = state.get(block, "pose_z")
            goal_achieved = bool(state.get(block, "goal_achieved"))
            color_r = state.get(block, "color_r")
            color_g = state.get(block, "color_g")
            color_b = state.get(block, "color_b")
            faceup = bool(state.get(block, "faceup"))  # True => arrow up, False => arrow down
            color = (color_r, color_g, color_b)

            # If "goal_achieved", we use a "+" hatch, otherwise if "held" is True, we might use "x", etc.
            # Just an example of hatch usage:
            held = ""  # For demonstration; replace with logic if relevant
            hatch = "+" if goal_achieved else ("x" if held else None)

            # Draw on x-z axis
            xz_rect = patches.Rectangle(
                (x - r, z - r),
                2*r,
                2*r,
                zorder=-y,  # so blocks at higher y don't occlude those behind
                hatch=hatch,
                linewidth=1,
                edgecolor='black',
                facecolor=color
            )
            xz_ax.add_patch(xz_rect)

            # Arrow for facing direction on xz
            xz_ax.annotate(
                "\u2191" if faceup else "\u2193",
                (x, z - r),
                textcoords="offset points",
                xytext=(0, 6),
                ha='center',
                va='bottom',
                fontsize=24
            )

            # Draw on y-z axis
            yz_rect = patches.Rectangle(
                (y - r, z - r),
                2*r,
                2*r,
                zorder=-x,
                hatch=hatch,
                linewidth=1,
                edgecolor='black',
                facecolor=color
            )
            yz_ax.add_patch(yz_rect)

            # Arrow for facing direction on yz
            yz_ax.annotate(
                "\u2191" if faceup else "\u2193",
                (y, z - r),
                textcoords="offset points",
                xytext=(0, 6),
                ha='center',
                va='bottom',
                fontsize=24
            )

        # --- 4) Create the bottom subplot: x-y with range 1.0 x 0.3 in data coords ---
        block_display_ax = fig.add_subplot(gs[1, :])  # Spans entire bottom row
        block_display_ax.set_xlabel("X", fontsize=14)
        block_display_ax.set_ylabel("Y", fontsize=14)

        # The user wants a 1.0 x 0.3 "box" in data coordinates
        block_display_ax.set_xlim(0, 1)
        block_display_ax.set_ylim(0, 0.3)

        # We'll not force aspect='equal' because the user specifically wants the data range to be [0..1]x[0..0.3].
        # If you do want the subplot to physically appear 1:0.3, you can play with figure or subplot size.

        # Each block is 0.1 x 0.1 in size, arranged evenly in the 1.0 range along x
        block_size = 0.1
        n_blocks = len(blocks)

        if n_blocks > 0:
            total_width = n_blocks * block_size
            leftover = 1.0 - total_width
            # The space to distribute between blocks is leftover, so define a gap on each side and between blocks.
            gap = leftover / (n_blocks + 1) if leftover > 0 else 0.0

            for i, block in enumerate(sorted(blocks)):
                # Compute x0 position
                x0 = gap*(i + 1) + i*block_size
                y0 = 0.1  # Put the square in the middle of the 0..0.3 range

                color_r = state.get(block, "color_r")
                color_g = state.get(block, "color_g")
                color_b = state.get(block, "color_b")
                irr_x = state.get(block, "irr_x")
                irr_x = (irr_x - self.blo_x_lb) / (self.blo_x_ub - self.blo_x_lb)
                irr_x = x0 + irr_x * block_size
                irr_y = state.get(block, "irr_y")
                irr_y = (irr_y - self.blo_y_lb) / (self.blo_y_ub - self.blo_y_lb)
                irr_y = y0 + irr_y * block_size
                irr_h = state.get(block, "irr_h")
                irr_h = round(irr_h, 3)
                block_color = (color_r, color_g, color_b)
                single = self._Single_holds(state, [block])

                # Draw the square patch for the block
                if single:
                    edge_color = "black"
                else:
                    edge_color = "red"
                block_display_ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        block_size,
                        block_size,
                        facecolor=block_color,
                        edgecolor=edge_color,
                        linewidth=2.0
                    )
                )

                # (5) In each block, add an "x" marker at some customized location (e.g., center)
                block_display_ax.text(
                    irr_x,
                    irr_y,
                    "x",
                    ha='center',
                    va='center',
                    fontsize=24,
                    color="black"
                )

                # (4) Above each block, add text about its name
                block_display_ax.text(
                    x0 + block_size/2,
                    y0 + block_size + 0.025,
                    str(block.name) + f"(H {irr_h})",
                    ha='center',
                    va='bottom',
                    fontsize=16,
                    color="black"
                )

        # Optional caption as figure title
        if caption is not None:
            plt.suptitle(caption, fontsize=20)

        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')

        return fig
    
class BlocksEngravePcdEnv(BlocksEngraveVecEnv):
    """BlocksOnClear domain."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._block_type = Type("block", [
            "pcd",
            "pose_x", "pose_y", "pose_z", 
            "irr_x", "irr_y", "irr_h",
            "color_r", "color_g",
            "color_b", "faceup", "goal_achieved"
        ]) # we don't need "held" in low-level features, should be predicates.
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "pose_z", "fingers"])
        # Predicates
        self._On = Predicate("On", [self._block_type, self._block_type],
                             self._On_holds)
        self._Matched = Predicate("Matched", [self._block_type, self._block_type],
                                    self._Matched_holds)
        self._Single = Predicate("Single", [self._block_type], self._Single_holds)
        self._FaceUp = Predicate("FaceUp", [self._block_type], self._FaceUp_holds)
        self._FaceDown = Predicate("FaceDown", [self._block_type], self._FaceDown_holds)
        self._OnTable = Predicate("OnTable", [self._block_type],
                                  self._OnTable_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._robot_type, self._block_type],
                                  self._Holding_holds) # Binary
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)
        self._NotEq = Predicate("NotEq", [self._block_type, self._block_type],
                                self._NotEq_holds)

        self._GoalAchieved = Predicate("GoalAchieved", 
                [self._block_type, self._block_type], self._GoalAchieved_holds)
        
        self._robot = Object("robby", self._robot_type)
        self.all_colors = [
            [0.0, 0.0, 1.0],  # blue
            [0.0, 1.0, 0.0],  # green
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
        self._num_blocks_train = CFG.blocks_engrave_num_blocks_train
        self._num_blocks_test = CFG.blocks_engrave_num_blocks_test

    @classmethod
    def get_name(cls) -> str:
        return "blocks_engrave_pcd"

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, fingers, rotate, eng_x, eng_y, eng_h, achieve_goal]
        lowers = np.array([self.x_lb, self.y_lb, 0.0, 0.0, \
                           0.0, self.blo_x_lb, self.blo_y_lb, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 2.0, 1.0, 
                           1.0, self.blo_x_ub, self.blo_y_ub, self.engrave_height, 1.0], dtype=np.float32)
        return Box(lowers, uppers)
    
    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._GoalAchieved}
    
    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnTable, self._GripperOpen, self._Holding,
            self._Clear, self._GoalAchieved, self._Matched, self._FaceUp, 
            self._FaceDown, self._Single, self._NotEq
        }
    
    def _gen_pcd_for_block(self, irr_x: float, irr_y: float, irr_h: float,
                           sigma: float) -> np.ndarray:
        from pytorch3d.ops import sample_points_from_meshes
        bumps = [
        {"A":  irr_h, "x0": irr_x, "y0": irr_y, "sigma": sigma},
        # {"A": -0.02, "x0": 0.07, "y0": 0.07, "sigma": 0.005},
        ]
        cube_minus_top_mesh, top_mesh = create_multi_bump_cube_components_pytorch3d(
            n=20, bumps=bumps, cube_size=self._block_size
        )
        
        # Sample many points on top
        num_top = int(CFG.blocks_engrave_num_points * 0.8)
        n_rest = CFG.blocks_engrave_num_points - num_top
        pc_top = sample_points_from_meshes(top_mesh, 
                                           num_samples=num_top)      # (1, n_top, 3)
        # Sample fewer points on the rest
        pc_rest = sample_points_from_meshes(cube_minus_top_mesh, num_samples=n_rest)  # (1, n_rest, 3)

        pc_top = pc_top.squeeze(0).cpu().numpy()
        pc_rest = pc_rest.squeeze(0).cpu().numpy()

        # Combine
        pcd = np.concatenate([pc_top, pc_rest], axis=0)

        # (Optional) Center the final point cloud or do any post-processing
        pcd -= self._block_size / 2

        return pcd
    
    def _transform_pcd(self, pcd: np.ndarray, x: float, y: float, z: float,
                    face_up: float) -> np.ndarray:
        """
        Translate the block's point cloud to (x,y,z) and 
        rotate 180 deg about the Y-axis if face_up == 0.0.
        """
        # 1) Optionally rotate by 180 deg about Y
        if face_up == 0.0:
            # Double-check that from_pitch(np.pi) indeed does a Y-rotation in your codebase.
            rot_quat = math_helpers.Quat.from_pitch(np.pi)
        else:
            rot_quat = math_helpers.Quat()  # Identity (no rotation)

        # Convert quaternion to a 3×3 rotation matrix
        rot_mat = rot_quat.to_matrix()  # shape: (3,3)

        # 2) Build the full 4×4 transformation matrix
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = np.array([x, y, z])

        # 3) Convert each point to homogeneous coords (x, y, z, 1) so we can multiply
        ones = np.ones((pcd.shape[0], 1), dtype=pcd.dtype)
        pcd_hom = np.hstack([pcd, ones])  # shape: (N,4)

        # 4) Apply the transformation
        pcd_transformed_hom = pcd_hom @ trans_mat.T  # shape: (N,4)
        pcd_transformed = pcd_transformed_hom[:, :3] # shape: (N,3)

        return pcd_transformed
    
    def _sample_state_from_piles(self, piles: List[List[Object]],
                                 rng: np.random.Generator) -> State:
        state_dict = {}
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
            r, g, b = self.all_colors[color_idx]
            # [pose_x, pose_y, pose_z, color_r, color_g, color_b, goal_achieved]
            # face_up = 1.0 if pile_j == 0 else rng.choice([0.0, 1.0])
            face_up = 1.0
            eng_x = rng.uniform(self.blo_x_lb, self.blo_x_ub)
            eng_y = rng.uniform(self.blo_y_lb, self.blo_y_ub)
            eng_h = self.engrave_height
            sigma = rng.uniform(CFG.blocks_engrave_sigma[0],
                                CFG.blocks_engrave_sigma[1])
            # initially all the surfaces are positive and facing up
            pcd = self._gen_pcd_for_block(eng_x, eng_y, eng_h, sigma)
            transformed_pcd = self._transform_pcd(pcd, x, y, z, face_up)
            state_dict[block] = {
                "pcd": transformed_pcd,
                "pose_x": x,
                "pose_y": y,
                "pose_z": z,
                "color_r": r,
                "color_g": g,
                "color_b": b,
                "faceup": face_up,
                "goal_achieved": 0.0,
                "irr_x": eng_x,
                "irr_y": eng_y,
                "irr_h": eng_h
            }
            color_idx += 1
        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = 1.0  # fingers start out open
        state_dict[self._robot] = {
            "pose_x": rx,
            "pose_y": ry,
            "pose_z": rz,
            "fingers": rf
        }
        return utils.create_state_from_dict(state_dict)
    
    def _sample_initial_piles(self, num_blocks: int,
                              rng: np.random.Generator) -> List[List[Object]]:
        piles: List[List[Object]] = []
        for block_num in range(num_blocks):
            block = Object(f"block{block_num}", self._block_type)
            # If coin flip, start new pile
            if (block_num == 0) or (rng.uniform() < 0.2) and len(piles) < 2:
                piles.append([])
            # Add block to pile
            piles[-1].append(block)
        return piles
    
    def _sample_goal_from_piles(self, num_blocks: int,
                                piles: List[List[Object]],
                                rng: np.random.Generator,
                                init_state: State) -> Tuple[Set[GroundAtom], State]:
        # Sample goal pile that is different from initial
        new_state = init_state.copy()
        while True:
            goal_piles: List[List[Object]] = [[]]
            for block_num in rng.permutation(range(num_blocks)):
                block = Object(f"block{block_num}", self._block_type)
                if len(goal_piles[-1]) == 2:
                    goal_piles.append([])
                # Add block to pile
                goal_piles[-1].append(block)
            if goal_piles != piles:
                break
        # Create goal from piles
        goal_atoms = set()
        for pile in goal_piles:
            if len(pile) > 1:
                goal_atoms.add(GroundAtom(self._GoalAchieved, [pile[-1], pile[-2]]))
                # if rng.random() < 0.5:
                #     # initially they are matched
                #     eng_x = rng.uniform(self.blo_x_lb, self.blo_x_ub)
                #     eng_y = rng.uniform(self.blo_y_lb, self.blo_y_ub)
                #     eng_h = self.engrave_height
                #     sigma = CFG.blocks_engrave_sigma
                #     new_pcd = self._gen_pcd_for_block(eng_x, eng_y, eng_h, sigma)
                #     top_block_x = init_state.get(pile[-1], "pose_x")
                #     top_block_y = init_state.get(pile[-1], "pose_y")
                #     top_block_z = init_state.get(pile[-1], "pose_z")
                #     face_up = init_state.get(pile[-1], "faceup")
                #     transformed_pcd = self._transform_pcd(new_pcd, top_block_x, \
                #             top_block_y, top_block_z, face_up)
                #     new_state.set(pile[-1], "pcd", transformed_pcd)
                #     new_state.set(pile[-1], "irr_x", eng_x)
                #     new_state.set(pile[-1], "irr_y", eng_y)
                #     new_state.set(pile[-1], "irr_h", eng_h)

                #     mirrored_x = self.blo_x_ub - eng_x + self.blo_x_lb
                #     new_pcd = self._gen_pcd_for_block(mirrored_x, eng_y, -eng_h, sigma)
                #     bottom_block_x = init_state.get(pile[-2], "pose_x")
                #     bottom_block_y = init_state.get(pile[-2], "pose_y")
                #     bottom_block_z = init_state.get(pile[-2], "pose_z")
                #     face_up = init_state.get(pile[-2], "faceup")
                #     transformed_pcd = self._transform_pcd(new_pcd, bottom_block_x, \
                #             bottom_block_y, bottom_block_z, face_up)
                #     new_state.set(pile[-2], "pcd", transformed_pcd)
                #     new_state.set(pile[-2], "irr_x", mirrored_x)
                #     new_state.set(pile[-2], "irr_y", eng_y)
                #     new_state.set(pile[-2], "irr_h", -eng_h)

                    # assert self._Matched_holds(new_state, [pile[-1], pile[-2]])
        return goal_atoms, new_state
    
    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks)
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            while True:  # repeat until goal is not satisfied
                # modify the init state by creating random single and matched blocks
                goal, new_state = self._sample_goal_from_piles(num_blocks, piles, rng, init_state)
                if not all(goal_atom.holds(new_state) for goal_atom in goal):
                    break
            tasks.append(EnvironmentTask(new_state, goal))
        return tasks
    
    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        x, y, z, fingers, rotate,\
        eng_x, eng_y, eng_h, achieve_goal = action.arr
        # Infer which transition function to follow
        if achieve_goal == 1:
            return self._transition_achievegoal(state, x, y, z)
        if rotate != 0:
            return self._transition_rotate(state, x, y, z, rotate)
        if eng_h > 0.0:
            return self._transition_engrave(state, 
                                            x, y, z,
                                            eng_x, eng_y, eng_h)
        if fingers < 0.5:
            return self._transition_pick(state, x, y, z)
        if z < self.table_height + self._block_size:
            return self._transition_putontable(state, x, y, z)
        return self._transition_stack(state, x, y, z)
    
    def _transition_engrave(self, state: State, x: float, y: float, z: float,
                            eng_x: float, eng_y: float, eng_h: float) -> State:
        next_state = state.copy()
        top_block = self._get_block_at_xyz(state, x, y, z)
        if top_block is None:  # no block at this pose
            return next_state
        if not self._block_is_clear(top_block, state):
            return next_state
        if not self._Single_holds(state, [top_block]):
            return next_state
        bottom_block = self._get_highest_block_below(state, x, y, z)
        if bottom_block is None:
            return next_state
        if not self._OnTable_holds(state, [bottom_block]):
            return next_state
        if not self._On_holds(state, [top_block, bottom_block]):
            return next_state
        if not self._Single_holds(state, [bottom_block]):
            return next_state
        next_state.set(top_block, "irr_x", eng_x)
        next_state.set(top_block, "irr_y", eng_y)
        next_state.set(top_block, "irr_h", eng_h)
        sigma = self._train_rng.uniform(CFG.blocks_engrave_sigma[0],
                                CFG.blocks_engrave_sigma[1])
        new_pcd = self._gen_pcd_for_block(eng_x, eng_y, eng_h, sigma)
        top_block_x = state.get(top_block, "pose_x")
        top_block_y = state.get(top_block, "pose_y")
        top_block_z = state.get(top_block, "pose_z")
        face_up = state.get(top_block, "faceup")
        transformed_pcd = self._transform_pcd(new_pcd, top_block_x, \
                            top_block_y, top_block_z, face_up)
        next_state.set(top_block, "pcd", transformed_pcd)
        # need to mirror the x coordinates
        next_state.set(bottom_block, "irr_y", eng_y)
        next_state.set(bottom_block, "irr_h", -eng_h)
        mirrored_x = self.blo_x_ub - eng_x + self.blo_x_lb
        next_state.set(bottom_block, "irr_x", mirrored_x)
        new_pcd = self._gen_pcd_for_block(mirrored_x, eng_y, -eng_h, sigma)
        bottom_block_x = state.get(bottom_block, "pose_x")
        bottom_block_y = state.get(bottom_block, "pose_y")
        bottom_block_z = state.get(bottom_block, "pose_z")
        face_up = state.get(bottom_block, "faceup")
        transformed_pcd = self._transform_pcd(new_pcd, bottom_block_x, \
                            bottom_block_y, bottom_block_z, face_up)
        next_state.set(bottom_block, "pcd", transformed_pcd)
        return next_state
    
    def _transition_rotate(self, state: State, x: float, y: float, z: float,
                            rotate: float) -> State:
        next_state = state.copy()
        block = self._get_block_at_xyz(state, x, y, z)
        if block is None:  # no block at this pose
            return next_state
        if not self._Holding_holds(state, [self._robot, block]):
            return next_state
        assert rotate > 0
        current_faceup = state.get(block, "faceup")
        next_state.set(block, "faceup", 1.0 - current_faceup)
        pcd = state.get(block, "pcd").copy()
        center = np.array([x, y, z])

        # 1) Translate point cloud so block center is at (0,0,0)
        pcd_centered = pcd - center

        rot_mat = math_helpers.Quat.from_pitch(np.pi).to_matrix()
        # Apply that rotation to each point
        pcd_rotated = pcd_centered @ rot_mat.T

        # 3) Translate back to the block's original center
        pcd_transformed = pcd_rotated + center

        # Update the state
        next_state.set(block, "pcd", pcd_transformed)

        return next_state
        
    def _transition_achievegoal(self, state: State, x: float, y: float, z: float) -> State:
        next_state = state.copy()
        top_block = self._get_block_at_xyz(state, x, y, z)
        if top_block is None:  # no block at this pose
            return next_state
        if not self._block_is_clear(top_block, state):
            return next_state
        if not self._FaceDown_holds(state, [top_block]):
            return next_state
        
        bottom_block = self._get_highest_block_below(state, x, y, z)
        if bottom_block is None:
            return next_state
        if not self._On_holds(state, [top_block, bottom_block]):
            return next_state
        if not self._Matched_holds(state, [top_block, bottom_block]):
            return next_state
        if not self._FaceUp_holds(state, [bottom_block]):
            return next_state
        next_state.set(top_block, "goal_achieved", 0.5)
        next_state.set(bottom_block, "goal_achieved", 1.0)
        return next_state
    
    def _transition_pick(self, state: State, x: float, y: float,
                         z: float) -> State:
        next_state = state.copy()
        # Can only pick if fingers are open
        if not self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_block_at_xyz(state, x, y, z)
        if block is None:  # no block at this pose
            return next_state
        # Can only pick if object is clear
        if not self._block_is_clear(block, state):
            return next_state
        # Execute pick
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        # Pick up block to a height of half the block size, to delete "On" in low level
        delta_z = self.pick_z - state.get(block, "pose_z")
        next_state.set(block, "pose_z", self.pick_z)
        next_state.set(self._robot, "fingers", 0.0)  # close fingers
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        next_state.set(self._robot, "pose_z", self.pick_z)
        # change pcd
        pcd = next_state.get(block, "pcd").copy()
        pcd[:, 2] += delta_z
        next_state.set(block, "pcd", pcd)
        return next_state
    
    def _transition_putontable(self, state: State, x: float, y: float,
                               z: float) -> State:
        next_state = state.copy()
        # Can only putontable if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_held_block(state)
        if block is None:
            return next_state
        # Check that table surface is clear at this pose
        poses = [[
            state.get(b, "pose_x"),
            state.get(b, "pose_y"),
            state.get(b, "pose_z")
        ] for b in state if b.is_instance(self._block_type)]
        existing_xys = {(float(p[0]), float(p[1])) for p in poses}
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        # Execute putontable
        delta_x = x - state.get(block, "pose_x")
        delta_y = y - state.get(block, "pose_y")
        delta_z = z - state.get(block, "pose_z")
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        next_state.set(self._robot, "pose_z", z)
        # change pcd
        pcd = next_state.get(block, "pcd").copy()
        pcd[:, 0] += delta_x
        pcd[:, 1] += delta_y
        pcd[:, 2] += delta_z
        next_state.set(block, "pcd", pcd)
        return next_state
    
    def _transition_stack(self, state: State, x: float, y: float,
                          z: float) -> State:
        next_state = state.copy()
        # Can only stack if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        # Check that both blocks exist
        block = self._get_held_block(state)
        if block is None:
            return next_state
        other_block = self._get_highest_block_below(state, x, y, z)
        if other_block is None:  # no block to stack onto
            return next_state
        # Can't stack onto yourself!
        if block == other_block:
            return next_state
        # Need block we're stacking onto to be clear
        if not self._block_is_clear(other_block, state):
            return next_state
        # Execute stack by snapping into place
        cur_x = state.get(other_block, "pose_x")
        cur_y = state.get(other_block, "pose_y")
        cur_z = state.get(other_block, "pose_z")
        delta_x = cur_x - state.get(block, "pose_x")
        delta_y = cur_y - state.get(block, "pose_y")
        delta_z = cur_z + self._block_size - state.get(block, "pose_z")
        next_state.set(block, "pose_x", cur_x)
        next_state.set(block, "pose_y", cur_y)
        next_state.set(block, "pose_z", cur_z + self._block_size)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers
        next_state.set(self._robot, "pose_x", cur_x)
        next_state.set(self._robot, "pose_y", cur_y)
        next_state.set(self._robot, "pose_z", cur_z + self._block_size)
        # change pcd
        pcd = next_state.get(block, "pcd").copy()
        pcd[:, 0] += delta_x
        pcd[:, 1] += delta_y
        pcd[:, 2] += delta_z
        next_state.set(block, "pcd", pcd)
        return next_state
    
    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
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
        return (desired_z-self.on_tol < z < desired_z+self.on_tol)
    
    def _GoalAchieved_holds(self, state: State, objects: Sequence[Object]) -> bool:
        top_block, bottom_block = objects
        goal_achieve1 = state.get(top_block, "goal_achieved")
        goal_achieve2 = state.get(bottom_block, "goal_achieved")
        order = goal_achieve1 < goal_achieve2
        achieved = goal_achieve1 > 0
        if order and achieved:
            return True
        return False
    
    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        for other_block in state:
            if other_block.type != self._block_type:
                continue
            if self._On_holds(state, [other_block, block]):
                return False
        return True
    
    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, block = objects
        x_r = state.get(robot, "pose_x")
        y_r = state.get(robot, "pose_y")
        z_r = state.get(robot, "pose_z")
        f_r = state.get(robot, "fingers")

        x_b = state.get(block, "pose_x")
        y_b = state.get(block, "pose_y")
        z_b = state.get(block, "pose_z")
        if f_r > 0.5:
            # gripper is open
            return False
        else:
            # gripper is closed
            return np.isclose([x_r, y_r, z_r], [x_b, y_b, z_b]).all()
        
    def _FaceUp_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "faceup") > 0.5
    
    def _FaceDown_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "faceup") < 0.5
    
    def _Matched_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        x_1 = state.get(block1, "irr_x")
        y_1 = state.get(block1, "irr_y")
        h_1 = state.get(block1, "irr_h")
        x_2 = state.get(block2, "irr_x")
        y_2 = state.get(block2, "irr_y")
        h_2 = state.get(block2, "irr_h")
        x_ok = np.isclose(x_1 + x_2, self.blo_x_ub + self.blo_x_lb)
        y_ok = np.isclose(y_1, y_2)
        h_ok = np.isclose(h_1, -h_2)
        is_on = self._On_holds(state, [block1, block2])
        return x_ok and y_ok and h_ok and is_on

    def _NotEq_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        block1_x = state.get(block1, "pose_x")
        block1_y = state.get(block1, "pose_y")
        block1_z = state.get(block1, "pose_z")
        block2_x = state.get(block2, "pose_x")
        block2_y = state.get(block2, "pose_y")
        block2_z = state.get(block2, "pose_z")
        dist3d = np.linalg.norm([block1_x-block2_x, block1_y-block2_y, block1_z-block2_z])
        return dist3d > self._block_size // 2
    
    def _Single_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        # the block is unique in this world, no match yet
        for other_block in state.get_objects(self._block_type):
            if self._Matched_holds(state, [block, other_block]):
                return False
        return True
        
    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            if self._Holding_holds(state, [self._robot, block]):
                return block
            
    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        """Create a task from a JSON file.

        By default, we assume JSON files are in the following format:

        {
            "objects": {
                <object name>: <type name>
            }
            "init": {
                <object name>: {
                    <feature name>: <value>
                }
            }
            "goal": {
                <predicate name> : [
                    [<object name>]
                ]
            }
        }

        Instead of "goal", "language_goal" can also be used.

        Environments can override this method to handle different formats.
        """
        with open(json_file, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        # Parse objects.
        type_name_to_type = {t.name: t for t in self.types}
        object_name_to_object: Dict[str, Object] = {}
        for obj_name, type_name in json_dict["objects"].items():
            obj_type = type_name_to_type[type_name]
            obj = Object(obj_name, obj_type)
            object_name_to_object[obj_name] = obj
        assert set(object_name_to_object).issubset(set(json_dict["init"])), \
            "The init state can only include objects in `objects`."
        assert set(object_name_to_object).issuperset(set(json_dict["init"])), \
            "The init state must include every object in `objects`."
        # Parse initial state.
        init_dict: Dict[Object, Dict[str, Array]] = {}
        for obj_name, obj_dict in json_dict["init"].items():
            obj = object_name_to_object[obj_name]
            init_dict[obj] = {}
            for k, v in obj_dict.items():
                if k == "pcd":
                    # convert list to numpy array
                    v = np.array(v, dtype=np.float32)
                init_dict[obj][k] = v
        init_state = utils.create_state_from_dict(init_dict)
        # Parse goal.
        if "goal" in json_dict:
            goal = self._parse_goal_from_json(json_dict["goal"],
                                              object_name_to_object)
        else:
            assert "language_goal" in json_dict
            goal = self._parse_language_goal_from_json(
                json_dict["language_goal"], object_name_to_object)
        return EnvironmentTask(init_state, goal)
    
    def render_state_plt(
        self,
        state,
        task,
        action: Optional[object] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        if CFG.blocks_engrave_render_mode == "3d":
            return self.render_state_plt_3d(state, task, action, save_path)
        else:
            return self.render_state_plt_2d(state, task, action, save_path)

    def render_state_plt_3d(self,
                            state: State,
                            task: EnvironmentTask,
                            action: Optional[Action] = None,
                            save_path: Optional[str] = None):
        # Set the image resolution
        width, height = 1490, 1080  # Modify as needed

        # Create an offscreen renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)  # type: ignore
        scene = renderer.scene

        # World coordinates (x, y, z axes)
        world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        scene.add_geometry(
            "world_frame",
            world_coordinate_frame,
            o3d.visualization.rendering.MaterialRecord(),  # type: ignore
        )

        # Create a ground plane (just a box in this example)
        plane = o3d.geometry.TriangleMesh.create_box(
            width=(self.x_ub - self.x_lb),
            height=(self.y_ub - self.y_lb),
            depth=0.01
        )
        plane_translation = np.array([self.x_lb, self.y_lb, self.table_height])
        plane.translate(plane_translation)

        ground_material = o3d.visualization.rendering.MaterialRecord()  # type: ignore
        ground_material.base_color = [0.55, 0.27, 0.07, 0.2]  # RGBA
        scene.add_geometry("plane", plane, ground_material)

        # 1) Collect all blocks' data (points + base color).
        blocks_data = []
        blocks = state.get_objects(self._block_type)
        for block in blocks:
            block_name = block.name
            block_pcd = state.get(block, "pcd")  # shape: (N, 3)
            # Retrieve color from state
            block_r = state.get(block, "color_r")
            block_g = state.get(block, "color_g")
            block_b = state.get(block, "color_b")

            # Normalize for some consistent hue (optional)
            color_arr = np.array([block_r, block_g, block_b], dtype=np.float32)
            denom = color_arr.sum()
            if denom > 1e-8:
                color_arr /= denom  # e.g. [r_norm, g_norm, b_norm]

            # -- Get block bounding-box info (assuming axis-aligned).
            #    For example, these might come from the State as well.
            size_x = self._block_size
            size_y = self._block_size
            size_z = self._block_size
            # Suppose you have the block's center (bx, by, bz) in the same frame
            bx = state.get(block, "pose_x")
            by = state.get(block, "pose_y")
            bz = state.get(block, "pose_z")

            # Define bounding box corners
            # For an axis-aligned box, we assume that
            # x_min = bx - size_x/2,  x_max = bx + size_x/2
            # y_min = by - size_y/2,  y_max = by + size_y/2
            # z_min = bz,             z_max = bz + size_z
            half_x = size_x / 2.0
            half_y = size_y / 2.0
            x_min, x_max = bx - half_x, bx + half_x
            y_min, y_max = by - half_y, by + half_y
            z_min, z_max = bz,         bz + size_z

            # We'll store per-point color in an (N, 3) array initially all = block base color
            pcd_colors = np.tile(color_arr.reshape(1, 3), (block_pcd.shape[0], 1))

            # -- Figure out which points are "on the surface" vs. "not on the surface."
            #    We'll define "on the surface" as points within a small epsilon of any
            #    bounding-box face. Everything else becomes red.
            epsilon = 1e-2

            # On any face if near x_min, x_max, y_min, y_max, z_min, or z_max
            on_face_mask = (
                (np.abs(block_pcd[:, 0] - x_min) < epsilon) |
                (np.abs(block_pcd[:, 0] - x_max) < epsilon) |
                (np.abs(block_pcd[:, 1] - y_min) < epsilon) |
                (np.abs(block_pcd[:, 1] - y_max) < epsilon) |
                (np.abs(block_pcd[:, 2] - z_min) < epsilon) |
                (np.abs(block_pcd[:, 2] - z_max) < epsilon)
            )

            # Everything else is not on the surface (either inside or way outside).
            not_on_face_mask = np.logical_not(on_face_mask)

            # Color these "non-surface" points red
            pcd_colors[not_on_face_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            blocks_data.append({
                "name": block_name,
                "points": block_pcd,
                "colors": pcd_colors,   # Nx3, with red for any point that's not on the bounding-box surface
                "base_color": color_arr # just for reference if needed
            })
                
        # 3) Now add the geometry to the scene with final colors
        name2info = {}
        for data in blocks_data:
            block_name = data["name"]
            block_points = data["points"]
            block_colors = data["colors"]

            # Build an Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(block_points)
            pcd.colors = o3d.utility.Vector3dVector(block_colors)

            # Create a material record so we can set transparency, point size, etc.
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = [1.0, 1.0, 1.0, 0.8]  
            mat.point_size = 2.0

            scene.add_geometry(block_name, pcd, mat)

            # For your name2info dict (used for legend)
            name2info[block_name] = {
                "color": data["base_color"].tolist(),  
            }

        # 4) Set up multiple camera views
        cam_x = (self.x_ub + self.x_lb) / 2
        cam_y = (self.y_ub + self.y_lb) / 2
        camera_positions = [
            [self.x_lb - 0.3, cam_y, 0.2],
            [self.x_lb - 0.3, cam_y, 0.9],
            [cam_x,       self.y_lb- 0.3, 0.2],
            [cam_x,       self.y_lb- 0.3, 0.9],
        ]
        up_directions = [
            [0.01, 0, 0.95],
            [0, 0, 1],
            [0.01, 0, 0.95],
            [0, 0, 1],
        ]

        rendered_images = []
        for i, camera_position in enumerate(camera_positions):
            camera = renderer.scene.camera
            # Look at the "center" of the table from each camera pos
            if camera_position[2] < 0.5:
                camera.look_at(
                    np.array([cam_x, cam_y, self.table_height]),
                    camera_position,
                    up_directions[i],
                )
            else:
                camera.look_at(
                    np.array([cam_x, cam_y, self.table_height + 0.8]),
                    camera_position,
                    up_directions[i],
                )
            image = renderer.render_to_image()
            image_np = np.asarray(image)
            rendered_images.append(image_np)

        # 5) Display the images in a single Matplotlib figure
        fig, axs = plt.subplots(2, 2, figsize=(90, 50))
        n = 0
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                ax.imshow(rendered_images[n])
                ax.axis("off")
                ax.set_title(f"View {n+1}")
                n += 1

        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.98, bottom=0.25, left=0.01, right=0.99)

        # 6) Create a legend for blocks (their *base* color).
        caption_ax = fig.add_axes([0.3, 0.01, 0.9, 0.1])  # bottom region
        caption_ax.axis("off")

        patches = []
        labels = []
        for name, info in name2info.items():
            color_patch = mpatches.Rectangle(
                (0, 0), 30, 30, facecolor=info["color"], edgecolor="none"
            )
            patches.append(color_patch)
            labels.append(f"{name}")

        caption_ax.legend(patches, labels, loc="center", fontsize=50, frameon=False, ncol=len(name2info))

        if save_path:
            fig.savefig(save_path, format="png", dpi=40)

        return fig

    def render_state_plt_2d(
        self,
        state,
        task,
        action: Optional[object] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Render state with two subplots for x-z, y-z, and a third x-y subplot for block display."""

        # --- 1) Create the figure and GridSpec for two rows, two columns ---
        #     Top row:    xz_ax (0,0), yz_ax (0,1)
        #     Bottom row: block_display_ax spans both columns (1,:)
        fig = plt.figure(figsize=(20, 14))
        gs = plt.GridSpec(
            2, 2,
            height_ratios=[6, 4],  # More space for the main row, less for the bottom
            wspace=0.3,
            hspace=0.3
        )

        # --- 2) Create the top-row subplots for x-z and y-z ---
        xz_ax = fig.add_subplot(gs[0, 0])
        yz_ax = fig.add_subplot(gs[0, 1])

        # Set up x-z axis
        r = self._block_size * 0.5
        xz_ax.set_xlabel("x", fontsize=18)
        xz_ax.set_ylabel("z", fontsize=18)
        xz_ax.set_xlim(self.x_lb - 2*r, self.x_ub + 2*r)
        xz_ax.set_ylim(self.table_height, self.table_height + r*20 + 0.1)

        # Set up y-z axis
        yz_ax.set_xlabel("y", fontsize=18)
        yz_ax.set_ylabel("z", fontsize=18)
        yz_ax.set_xlim(self.y_lb - 2*r, self.y_ub + 2*r)
        yz_ax.set_ylim(self.table_height, self.table_height + r*20 + 0.1)

        # --- 3) Collect and draw blocks on top subplots ---
        blocks = [o for o in state if o.is_instance(self._block_type)]

        for block in sorted(blocks):
            x = state.get(block, "pose_x")
            y = state.get(block, "pose_y")
            z = state.get(block, "pose_z")
            goal_achieved = bool(state.get(block, "goal_achieved"))
            color_r = state.get(block, "color_r")
            color_g = state.get(block, "color_g")
            color_b = state.get(block, "color_b")
            faceup = bool(state.get(block, "faceup"))  # True => arrow up, False => arrow down
            color = (color_r, color_g, color_b)

            # If "goal_achieved", we use a "+" hatch, otherwise if "held" is True, we might use "x", etc.
            # Just an example of hatch usage:
            held = ""  # For demonstration; replace with logic if relevant
            hatch = "+" if goal_achieved else ("x" if held else None)

            # Draw on x-z axis
            xz_rect = patches.Rectangle(
                (x - r, z - r),
                2*r,
                2*r,
                zorder=-y,  # so blocks at higher y don't occlude those behind
                hatch=hatch,
                linewidth=1,
                edgecolor='black',
                facecolor=color
            )
            xz_ax.add_patch(xz_rect)

            # Arrow for facing direction on xz
            xz_ax.annotate(
                "\u2191" if faceup else "\u2193",
                (x, z - r),
                textcoords="offset points",
                xytext=(0, 6),
                ha='center',
                va='bottom',
                fontsize=24
            )

            # Draw on y-z axis
            yz_rect = patches.Rectangle(
                (y - r, z - r),
                2*r,
                2*r,
                zorder=-x,
                hatch=hatch,
                linewidth=1,
                edgecolor='black',
                facecolor=color
            )
            yz_ax.add_patch(yz_rect)

            # Arrow for facing direction on yz
            yz_ax.annotate(
                "\u2191" if faceup else "\u2193",
                (y, z - r),
                textcoords="offset points",
                xytext=(0, 6),
                ha='center',
                va='bottom',
                fontsize=24
            )

        # --- 4) Create the bottom subplot: x-y with range 1.0 x 0.3 in data coords ---
        block_display_ax = fig.add_subplot(gs[1, :])  # Spans entire bottom row
        block_display_ax.set_xlabel("X", fontsize=14)
        block_display_ax.set_ylabel("Y", fontsize=14)

        # The user wants a 1.0 x 0.3 "box" in data coordinates
        block_display_ax.set_xlim(0, 1)
        block_display_ax.set_ylim(0, 0.3)

        # We'll not force aspect='equal' because the user specifically wants the data range to be [0..1]x[0..0.3].
        # If you do want the subplot to physically appear 1:0.3, you can play with figure or subplot size.

        # Each block is 0.1 x 0.1 in size, arranged evenly in the 1.0 range along x
        block_size = 0.1
        n_blocks = len(blocks)

        if n_blocks > 0:
            total_width = n_blocks * block_size
            leftover = 1.0 - total_width
            # The space to distribute between blocks is leftover, so define a gap on each side and between blocks.
            gap = leftover / (n_blocks + 1) if leftover > 0 else 0.0

            for i, block in enumerate(sorted(blocks)):
                # Compute x0 position
                x0 = gap*(i + 1) + i*block_size
                y0 = 0.1  # Put the square in the middle of the 0..0.3 range

                color_r = state.get(block, "color_r")
                color_g = state.get(block, "color_g")
                color_b = state.get(block, "color_b")
                irr_x = state.get(block, "irr_x")
                irr_x = x0 + irr_x
                irr_y = state.get(block, "irr_y")
                irr_y = y0 + irr_y
                irr_h = state.get(block, "irr_h")
                irr_h = round(irr_h, 3)
                block_color = (color_r, color_g, color_b)
                single = self._Single_holds(state, [block])

                # Draw the square patch for the block
                if single:
                    edge_color = "black"
                else:
                    edge_color = "red"
                block_display_ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        block_size,
                        block_size,
                        facecolor=block_color,
                        edgecolor=edge_color,
                        linewidth=2.0
                    )
                )

                # (5) In each block, add an "x" marker at some customized location (e.g., center)
                block_display_ax.text(
                    irr_x,
                    irr_y,
                    "x",
                    ha='center',
                    va='center',
                    fontsize=24,
                    color="black"
                )

                # (4) Above each block, add text about its name
                block_display_ax.text(
                    x0 + block_size/2,
                    y0 + block_size + 0.025,
                    str(block.name) + f"(H {irr_h})",
                    ha='center',
                    va='bottom',
                    fontsize=16,
                    color="black"
                )

        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')

        return fig