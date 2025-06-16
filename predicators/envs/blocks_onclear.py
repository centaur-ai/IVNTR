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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
import json
from typing import Sequence, Set, List, Dict, Tuple, Optional

from predicators.envs.blocks import BlocksEnv
from predicators import utils
import numpy as np
from gym.spaces import Box
from predicators.structs import Object, Predicate, State, Type, Action, Array, \
    GroundAtom, EnvironmentTask


class BlocksOnClearEnv(BlocksEnv):
    """BlocksOnClear domain."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Types
        self._block_type = Type("block", [
            "pose_x", "pose_y", "pose_z", "color_r", "color_g",
            "color_b", "goal_achieved"
        ]) # we don't need "held" in low-level features, should be predicates.
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "pose_z", "fingers"])
        # Predicates
        self._On = Predicate("On", [self._block_type, self._block_type],
                             self._On_holds)
        self._OnTable = Predicate("OnTable", [self._block_type],
                                  self._OnTable_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._robot_type, self._block_type],
                                  self._Holding_holds) # Binary
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)

        self._GoalAchieved = Predicate("GoalAchieved", 
                [self._block_type, self._block_type], self._GoalAchieved_holds)

    @classmethod
    def get_name(cls) -> str:
        return "blocks_onclear"

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
        init_dict: Dict[Object, Dict[str, float]] = {}
        for obj_name, obj_dict in json_dict["init"].items():
            obj = object_name_to_object[obj_name]
            init_dict[obj] = obj_dict.copy()
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
    
    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, fingers, achieve_goal]
        lowers = np.array([self.x_lb, self.y_lb, 0.0, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 10.0, 1.0, 1.0], dtype=np.float32)
        return Box(lowers, uppers)
    
    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._GoalAchieved}
    
    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnTable, self._GripperOpen, self._Holding,
            self._Clear, self._GoalAchieved
        }
    
    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        x, y, z, fingers, achieve_goal = action.arr
        # Infer which transition function to follow
        if achieve_goal == 1:
            return self._transition_achievegoal(state, x, y, z)
        if fingers < 0.5:
            return self._transition_pick(state, x, y, z)
        if z < self.table_height + self._block_size:
            return self._transition_putontable(state, x, y, z)
        return self._transition_stack(state, x, y, z)
    
    def _transition_achievegoal(self, state: State, x: float, y: float, z: float) -> State:
        next_state = state.copy()
        top_block = self._get_block_at_xyz(state, x, y, z)
        if top_block is None:  # no block at this pose
            return next_state
        if not self._block_is_clear(top_block, state):
            return next_state
        bottom_block = self._get_highest_block_below(state, x, y, z)
        if bottom_block is None:
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
    
    def _sample_state_from_piles(self, piles: List[List[Object]],
                                 rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
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
        for block, pile_idx in block_to_pile_idx.items():
            pile_i, pile_j = pile_idx
            x, y = pile_to_xy[pile_i]
            z = self.table_height + self._block_size * (0.5 + pile_j)
            r, g, b = rng.uniform(size=3)
            # [pose_x, pose_y, pose_z, color_r, color_g, color_b, goal_achieved]
            data[block] = np.array([x, y, z, r, g, b, 0.0])
        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = 1.0  # fingers start out open
        data[self._robot] = np.array([rx, ry, rz, rf], dtype=np.float32)
        return State(data)
    
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
                                rng: np.random.Generator) -> Set[GroundAtom]:
        # Sample goal pile that is different from initial
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
        return goal_atoms
    
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
        
    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            if self._Holding_holds(state, [self._robot, block]):
                return block
            
    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        r = self._block_size * 0.5  # block radius

        width_ratio = max(
            1. / 5,
            min(
                5.,  # prevent from being too extreme
                (self.y_ub - self.y_lb) / (self.x_ub - self.x_lb)))
        fig, (xz_ax, yz_ax) = plt.subplots(
            1,
            2,
            figsize=(20, 8),
            gridspec_kw={'width_ratios': [1, width_ratio]})
        xz_ax.set_xlabel("x", fontsize=24)
        xz_ax.set_ylabel("z", fontsize=24)
        xz_ax.set_xlim((self.x_lb - 2 * r, self.x_ub + 2 * r))
        xz_ax.set_ylim((self.table_height, self.table_height + r * 20 + 0.1))
        yz_ax.set_xlabel("y", fontsize=24)
        yz_ax.set_ylabel("z", fontsize=24)
        yz_ax.set_xlim((self.y_lb - 2 * r, self.y_ub + 2 * r))
        yz_ax.set_ylim((self.table_height, self.table_height + r * 20 + 0.1))

        blocks = [o for o in state if o.is_instance(self._block_type)]
        
        for block in sorted(blocks):
            x = state.get(block, "pose_x")
            y = state.get(block, "pose_y")
            z = state.get(block, "pose_z")
            goal_achieved = bool(state.get(block, "goal_achieved"))
            # RGB values are between 0 and 1.
            color_r = state.get(block, "color_r")
            color_g = state.get(block, "color_g")
            color_b = state.get(block, "color_b")
            color = (color_r, color_g, color_b)
            held = ''

            # xz axis
            if goal_achieved:
                xz_rect = patches.Rectangle((x - r, z - r),
                                            2 * r,
                                            2 * r,
                                            zorder=-y,
                                            hatch="+",
                                            linewidth=1,
                                            edgecolor='black',
                                            facecolor=color)
                xz_ax.add_patch(xz_rect)

                # yz axis
                yz_rect = patches.Rectangle((y - r, z - r),
                                            2 * r,
                                            2 * r,
                                            zorder=-x,
                                            hatch="+",
                                            linewidth=1,
                                            edgecolor='black',
                                            facecolor=color)
                yz_ax.add_patch(yz_rect)
            elif held:
                xz_rect = patches.Rectangle((x - r, z - r),
                                            2 * r,
                                            2 * r,
                                            zorder=-y,
                                            hatch="x",
                                            linewidth=1,
                                            edgecolor='black',
                                            facecolor=color)
                xz_ax.add_patch(xz_rect)

                # yz axis
                yz_rect = patches.Rectangle((y - r, z - r),
                                            2 * r,
                                            2 * r,
                                            zorder=-x,
                                            hatch="x",
                                            linewidth=1,
                                            edgecolor='black',
                                            facecolor=color)
                yz_ax.add_patch(yz_rect)
            else:
                xz_rect = patches.Rectangle((x - r, z - r),
                                            2 * r,
                                            2 * r,
                                            zorder=-y,
                                            linewidth=1,
                                            edgecolor='black',
                                            facecolor=color)
                xz_ax.add_patch(xz_rect)

                # yz axis
                yz_rect = patches.Rectangle((y - r, z - r),
                                            2 * r,
                                            2 * r,
                                            zorder=-x,
                                            linewidth=1,
                                            edgecolor='black',
                                            facecolor=color)
                yz_ax.add_patch(yz_rect)

        title = f"Held: {held}"
        if caption is not None:
            title += f"; {caption}"
        plt.suptitle(title, fontsize=24, wrap=True)
        plt.tight_layout()
        return fig