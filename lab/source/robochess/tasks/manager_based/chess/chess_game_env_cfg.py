"""Two arms facing each other across a board, one playing white and one playing black.

Separate from :mod:`franka_chess_env_cfg` rather than an extension of it, because
almost everything that makes the single-arm task work is singular: one ``robot``
entity, one ``ee_frame``, one arm action, and a command term that samples a random
piece and a random destination. A game has none of those -- two of each arm-shaped
thing, and moves chosen by the rules rather than sampled.

Geometry. The board sits ``board_distance`` in front of white, and black is placed a
further ``board_distance`` beyond it, yawed 180 degrees so it faces back across the
board. The two arms may be different models; each contributes its own reach and
mounting distance, so an asymmetric pair still ends up with the board between them.

.. code-block:: text

           white base            board centre            black base
    x =  base_x  ------------->  +white.board_distance  ------------->  +black.board_distance
    yaw    0                                                                180 deg

Success is not a termination term here. Whether a game ended in checkmate, in a draw
or by hitting the ply cap is known to the rules engine driving the episode, not to
anything observable in the scene, so the generator flags it through the recorder API
(:meth:`RecorderManager.set_success_to_episodes`) instead.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .board import PLAYABLE_SCENARIOS, BoardLayout, make_layout
from .franka_chess_env_cfg import (
    CAPTURE_TRAY_PITCH,
    CAPTURE_TRAY_SHAPE,
    DEFAULT_BOARD_SCALE,
    PIECE_SCALE_TAG,
    TABLE_TOP_Z,
    capture_tray_center,
    configure_chess_scene,
)
from .robot_configs import CHESS_ROBOT_OPTIONS, CHESS_ROBOTS, ChessRobotSpec

GAME_TABLE_SIZE = (1.90, 1.10, TABLE_TOP_Z)
"""Longer than the single-arm table: it has to reach under both bases."""

PLAYERS = ("white", "black")

FACING_HOME_JOINT_POS = {
    # Two arms facing each other sit ``2 x board_distance`` apart -- 0.90 m for a pair of
    # Frankas, each with 0.68 m of reach, so their workspaces overlap heavily across the
    # middle. The single-arm home pose reaches *forward* over the board, which puts both
    # wrists in the same place and has them resting against each other before the game
    # even starts. These postures fold each arm back over its own base instead.
    "franka": {
        "panda_joint1": 0.0,
        "panda_joint2": -1.05,
        "panda_joint3": 0.0,
        "panda_joint4": -2.62,
        "panda_joint5": 0.0,
        "panda_joint6": 1.72,
        "panda_joint7": 0.785,
        "panda_finger_joint.*": 0.04,
    },
}
"""Per-arm rest posture for the facing-pair layout, falling back to the single-arm home."""


@configclass
class ChessGameSceneCfg(InteractiveSceneCfg):
    """Table, two arms and a board. Pieces are attached by :func:`configure_chess_scene`."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=GAME_TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.18, 0.12), roughness=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP_Z / 2.0)),
    )

    # All four are filled in by ChessGameEnvCfg.set_players().
    robot_white: ArticulationCfg = MISSING
    robot_black: ArticulationCfg = MISSING
    ee_frame_white: FrameTransformerCfg = MISSING
    ee_frame_black: FrameTransformerCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.92, 0.92, 0.92), intensity=2600.0),
    )


@configclass
class ChessGameActionsCfg:
    """One task-space pose and one binary gripper per player."""

    arm_action_white: ActionTerm = MISSING
    gripper_action_white: ActionTerm = MISSING
    arm_action_black: ActionTerm = MISSING
    gripper_action_black: ActionTerm = MISSING


@configclass
class ChessGameObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        joint_pos_white = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_white")})
        joint_vel_white = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_white")})
        eef_pos_white = ObsTerm(func=mdp.ee_frame_pos, params={"ee_frame_cfg": SceneEntityCfg("ee_frame_white")})
        eef_quat_white = ObsTerm(func=mdp.ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("ee_frame_white")})
        gripper_white = ObsTerm(
            func=mdp.gripper_open_fraction,
            params={"open_command": {}, "close_command": {}, "robot_cfg": SceneEntityCfg("robot_white")},
        )

        joint_pos_black = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_black")})
        joint_vel_black = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_black")})
        eef_pos_black = ObsTerm(func=mdp.ee_frame_pos, params={"ee_frame_cfg": SceneEntityCfg("ee_frame_black")})
        eef_quat_black = ObsTerm(func=mdp.ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("ee_frame_black")})
        gripper_black = ObsTerm(
            func=mdp.gripper_open_fraction,
            params={"open_command": {}, "close_command": {}, "robot_cfg": SceneEntityCfg("robot_black")},
        )

        piece_positions = ObsTerm(func=mdp.piece_positions, params={"piece_names": []})
        piece_orientations = ObsTerm(func=mdp.piece_orientations, params={"piece_names": []})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ChessGameEventCfg:
    reset_white_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot_white")},
    )
    reset_black_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot_black")},
    )
    # A game starts from the book position, so unlike the pick-and-place task the
    # pieces are placed exactly rather than jittered.
    reset_pieces = EventTerm(
        func=mdp.reset_pieces_on_board,
        mode="reset",
        params={"piece_names": [], "position_noise": 0.0, "yaw_noise": 0.0},
    )


@configclass
class ChessGameTerminationsCfg:
    """Only the clock. Whether the *game* is over is a rules question, not a scene one."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ChessGameEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-vs-robot chess on one of :data:`~robochess.tasks.manager_based.chess.board.PLAYABLE_SCENARIOS`."""

    white_robot: str = "franka"
    black_robot: str = "franka"
    chess_scenario: str = "3x3"
    piece_scale_tag: str = PIECE_SCALE_TAG
    board_scale: float | None = None

    max_plies: int = 12
    """Ply cap. Games that run past it are recorded as unfinished rather than truncated
    mid-move, so the ply budget also sets :attr:`episode_length_s`."""

    seconds_per_ply: float = 26.0
    """Worst-case clock for a *single* pick-and-place: 10.3 s of interpolation plus up
    to 2 s of settling on each of the eight phases."""

    CAPTURE_FACTOR = 2
    """A capture is two pick-and-places -- clear the taken piece to the tray, then move
    the capturing piece -- so a game of nothing but captures needs twice the budget.
    Sizing for the worst case matters because a game that runs out of clock is thrown
    away whole, however many plies it had already played correctly."""

    scene: ChessGameSceneCfg = ChessGameSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    observations: ChessGameObservationsCfg = ChessGameObservationsCfg()
    actions: ChessGameActionsCfg = ChessGameActionsCfg()
    events: ChessGameEventCfg = ChessGameEventCfg()
    terminations: ChessGameTerminationsCfg = ChessGameTerminationsCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = self.episode_budget()
        self.viewer.eye = (0.0, 1.60, 1.65)
        self.viewer.lookat = (0.0, 0.0, 0.80)
        self.set_players(self.white_robot, self.black_robot)
        self.set_chess_scenario(self.chess_scenario)

    """
    Operations.
    """

    def episode_budget(self) -> float:
        """Episode clock, in seconds, allowing every ply to be a capture."""
        return self.max_plies * self.seconds_per_ply * self.CAPTURE_FACTOR

    def player_spec(self, player: str) -> ChessRobotSpec:
        return CHESS_ROBOTS[self.white_robot if player == "white" else self.black_robot]

    def chess_layout(self) -> BoardLayout:
        return make_layout(self.chess_scenario, self._resolved_board_scale(self.chess_scenario))

    def _resolved_board_scale(self, scenario: str) -> float:
        return self.board_scale if self.board_scale is not None else DEFAULT_BOARD_SCALE[scenario]

    def board_center(self) -> tuple[float, float]:
        """Board centre in the environment frame.

        Placed at the origin so the table, which has to span both arms, stays centred
        no matter how far apart the pair happens to need to sit.
        """
        return 0.0, 0.0

    def base_pose(self, player: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Where a player's arm is bolted down, and which way it faces.

        White keeps its own mounting height and lateral offset but is pushed back to
        put the board centre at the origin; black is the mirror image, yawed 180 deg.
        """
        spec = self.player_spec(player)
        sign = -1.0 if player == "white" else 1.0
        pos = (sign * spec.board_distance, spec.base_pos[1], spec.base_pos[2])
        # Isaac Lab 3.0 quaternions are (x, y, z, w); a 180 deg yaw is (0, 0, 1, 0).
        rot = (0.0, 0.0, 0.0, 1.0) if player == "white" else (0.0, 0.0, 1.0, 0.0)
        return pos, rot

    def set_players(self, white_robot: str, black_robot: str) -> None:
        """Configure both arms, their end-effector frames and their action terms."""
        for name in (white_robot, black_robot):
            if name not in CHESS_ROBOTS:
                raise ValueError(f"Unknown robot '{name}'. Choose from {CHESS_ROBOT_OPTIONS}.")
        self.white_robot, self.black_robot = white_robot, black_robot

        for player in PLAYERS:
            spec = self.player_spec(player)
            entity = f"robot_{player}"
            pos, rot = self.base_pose(player)

            init_state = spec.articulation.init_state.replace(pos=pos, rot=rot)
            home = FACING_HOME_JOINT_POS.get(spec.key, spec.home_joint_pos)
            if home:
                init_state = init_state.replace(joint_pos=dict(home))
            setattr(
                self.scene,
                entity,
                spec.articulation.replace(prim_path=f"{{ENV_REGEX_NS}}/Robot_{player}", init_state=init_state),
            )
            setattr(
                self.scene,
                f"ee_frame_{player}",
                FrameTransformerCfg(
                    prim_path=spec.base_prim_path(f"{{ENV_REGEX_NS}}/Robot_{player}"),
                    debug_vis=False,
                    visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path=f"/Visuals/EEFrame_{player}"),
                    target_frames=[
                        # Index 0 is the TCP, which is what every mdp term reads.
                        FrameTransformerCfg.FrameCfg(
                            prim_path=spec.ee_prim_path(f"{{ENV_REGEX_NS}}/Robot_{player}"),
                            name="end_effector",
                            offset=OffsetCfg(pos=spec.tcp_offset),
                        ),
                        FrameTransformerCfg.FrameCfg(
                            prim_path=spec.ee_prim_path(f"{{ENV_REGEX_NS}}/Robot_{player}"), name="ee_body"
                        ),
                    ],
                ),
            )

            setattr(
                self.actions,
                f"arm_action_{player}",
                DifferentialInverseKinematicsActionCfg(
                    asset_name=entity,
                    joint_names=list(spec.arm_joints),
                    body_name=spec.ee_body,
                    controller=DifferentialIKControllerCfg(
                        command_type="pose", use_relative_mode=False, ik_method="dls"
                    ),
                ),
            )
            setattr(
                self.actions,
                f"gripper_action_{player}",
                mdp.BinaryJointPositionActionCfg(
                    asset_name=entity,
                    joint_names=list(spec.gripper_joints),
                    open_command_expr=dict(spec.open_command),
                    close_command_expr=dict(spec.close_command),
                ),
            )

            event = getattr(self.events, f"reset_{player}_joints")
            event.params["asset_cfg"] = SceneEntityCfg(entity, joint_names=list(spec.arm_joints))

            gripper = getattr(self.observations.policy, f"gripper_{player}")
            gripper.params.update(
                {"open_command": dict(spec.open_command), "close_command": dict(spec.close_command)}
            )

        self.scene.table.init_state.pos = (0.0, 0.0, TABLE_TOP_Z / 2.0)
        if hasattr(self.scene, "chessboard"):
            self.set_chess_scenario(self.chess_scenario)

    def set_chess_scenario(self, scenario: str, board_scale: float | None = None) -> None:
        if scenario not in PLAYABLE_SCENARIOS:
            raise ValueError(
                f"'{scenario}' has no rule set. Playable scenarios are {PLAYABLE_SCENARIOS}."
            )
        self.chess_scenario = scenario
        if board_scale is not None:
            self.board_scale = board_scale
        layout = make_layout(scenario, self._resolved_board_scale(scenario))
        configure_chess_scene(self.scene, layout, self.piece_scale_tag, self.board_center())

        # configure_chess_scene lays down the single-arm task's shared tray; replace it
        # with one per player, each on its own side.
        if hasattr(self.scene, "capture_tray"):
            delattr(self.scene, "capture_tray")
        for player in PLAYERS:
            tray_x, tray_y = self.capture_tray_center(layout, player)
            setattr(
                self.scene,
                f"capture_tray_{player}",
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/capture_tray_{player}",
                    spawn=sim_utils.CuboidCfg(
                        size=(
                            CAPTURE_TRAY_SHAPE[0] * CAPTURE_TRAY_PITCH,
                            CAPTURE_TRAY_SHAPE[1] * CAPTURE_TRAY_PITCH,
                            0.001,
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.45, 0.42, 0.38), roughness=0.8
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(tray_x, tray_y, TABLE_TOP_Z + 0.0005)),
                ),
            )

        piece_names = [spec.name for spec in layout.pieces]
        self.observations.policy.piece_positions.params["piece_names"] = piece_names
        self.observations.policy.piece_orientations.params["piece_names"] = piece_names
        self.events.reset_pieces.params["piece_names"] = piece_names

    """
    Geometry helpers used by the game driver.
    """

    def board_square_pos(self, board_x: float, board_y: float) -> tuple[float, float, float]:
        center = self.board_center()
        return center[0] + board_x, center[1] + board_y, TABLE_TOP_Z

    def square_pos(self, layout: BoardLayout, file: int, rank: int) -> tuple[float, float, float]:
        return self.board_square_pos(*layout.square_center(file, rank))

    def capture_tray_center(self, layout: BoardLayout, player: str) -> tuple[float, float]:
        """Centre of ``player``'s own capture tray.

        One shared tray does not work here. The single-arm task puts it clear of the
        board's -y edge, which on the 1D board (eight cells, 672 mm) lands 0.70 m from
        a base with 0.68 m of reach -- so every capture aborts. Giving each player a
        tray pulled back towards its own base keeps it inside the workspace on every
        layout, and is what a human playing this would do anyway.
        """
        _, edge_y = capture_tray_center(layout, self.board_center())
        base, _ = self.base_pose(player)
        return base[0] / 2.0, edge_y

    def capture_tray_positions(self, layout: BoardLayout, player: str) -> list[tuple[float, float, float]]:
        columns, rows = CAPTURE_TRAY_SHAPE
        center_x, center_y = self.capture_tray_center(layout, player)
        return [
            (
                center_x + (col - (columns - 1) / 2.0) * CAPTURE_TRAY_PITCH,
                center_y + (row - (rows - 1) / 2.0) * CAPTURE_TRAY_PITCH,
                TABLE_TOP_Z,
            )
            for col in range(columns)
            for row in range(rows)
        ]

    def unreachable_squares(self, layout: BoardLayout) -> dict[str, list[tuple[int, int]]]:
        """Squares each player cannot comfortably reach, for reporting before a run.

        A game is only playable if both arms can reach every square either of them
        might have to touch, so this is worth checking up front rather than
        discovering it as a stalled episode 40 plies in.
        """
        out: dict[str, list[tuple[int, int]]] = {}
        for player in PLAYERS:
            spec = self.player_spec(player)
            base, _ = self.base_pose(player)
            far = []
            for rank in range(layout.num_ranks):
                for file in range(layout.num_files):
                    x, y, _ = self.square_pos(layout, file, rank)
                    if math.hypot(x - base[0], y - base[1]) > spec.reach:
                        far.append((file, rank))
            out[player] = far
        return out

    def unreachable_tray_slots(self, layout: BoardLayout) -> dict[str, int]:
        """How many of each player's own tray slots sit outside its reach.

        Checked separately from the squares because it is the failure that actually
        bit: every board square was reachable on the 1D layout while the tray was not,
        so a game aborted on its first capture with nothing in the pre-flight warning.
        """
        counts = {}
        for player in PLAYERS:
            spec = self.player_spec(player)
            base, _ = self.base_pose(player)
            counts[player] = sum(
                math.hypot(x - base[0], y - base[1]) > spec.reach * 0.98
                for x, y, _ in self.capture_tray_positions(layout, player)
            )
        return counts
