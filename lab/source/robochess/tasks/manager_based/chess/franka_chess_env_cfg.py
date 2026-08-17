"""Franka Panda chess pick-and-place environment.

A Franka Emika Panda stands at the edge of a table with a chess board in front of
it.  Every episode the :class:`~robochess.tasks.manager_based.chess.mdp.commands.ChessMoveCommand`
picks one piece and one destination -- an empty square, or a slot in the capture
tray beside the board.  The episode succeeds when that piece is standing upright
and settled there, out of the gripper, with the rest of the board still standing.

Unlike :mod:`robochess.tasks.manager_based.chess.chess_env_cfg` -- which spawns
render-only pieces for visual inspection -- the pieces here are rigid bodies built
by ``lab/scripts/prepare_chess_assets.py``: scaled up so a 80 mm Franka gripper can
close on them, with convex-decomposition collision that preserves the neck of each
piece.  Grasps for those exact meshes come from GraspGen (see
``lab/scripts/graspgen_chess_grasps.py``).
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from .board import BoardLayout, PieceSpec, make_layout
from .robot_configs import CHESS_ROBOT_OPTIONS, CHESS_ROBOTS, ChessRobotSpec

##
# Table / robot layout. The table top is the reference plane for everything else.
##

TABLE_TOP_Z = 0.77
TABLE_SIZE = (1.30, 1.10, TABLE_TOP_Z)
TABLE_CENTER = (0.15, 0.0)
TABLE_EDGE_MARGIN = 0.06
"""Keep destinations this far inside the table edge, so a placed piece cannot topple off."""

ROBOT_BASE_POS = (-0.23, 0.0, TABLE_TOP_Z)
BOARD_CENTER = (0.22, 0.0)
"""Legacy default board centre. The live value is ``ChessPickEnvCfg.board_center()``,
which places the board ``ChessRobotSpec.board_distance`` in front of whichever arm is
selected -- these robots differ by more than 2x in reach."""

DEFAULT_BOARD_SCALE = {"pieces": 1.4, "1d": 1.4, "3x3": 1.4, "4x4": 1.4, "minichess": 1.4, "8x8": 1.0}
"""How much to stretch the squares, per scenario.

At scale 1.0 the 1.5x pieces fill 78-98% of a 60 mm square: a king and its
neighbour are ~1 mm apart, so there is nowhere for a 22 mm finger to go and the
gripper knocks pieces over on the way in. 1.4 opens that to ~25 mm. The 8x8 board
stays at 1.0 because stretching a 0.48 m board by 1.4 pushes most of it outside
the Franka's reach.
"""

CAPTURE_TRAY_SHAPE = (2, 3)
CAPTURE_TRAY_PITCH = 0.07
CAPTURE_TRAY_GAP = 0.09
"""Where captured pieces go, parked beside the board with this much clearance.

The full 4x4 setup leaves no empty square, so the tray is the only legal
destination there; on the bigger boards it doubles the variety of place motions
in the dataset."""

PIECE_SCALE_TAG = "s150"
"""Which bake of ``lab/scripts/prepare_chess_assets.py`` to load (1.5x scale)."""

GENERATED_ASSET_DIR = Path(__file__).resolve().parents[6] / "assets" / "chess" / "generated"

PIECE_COLORS = {
    "white": (0.90, 0.88, 0.82),
    "black": (0.06, 0.06, 0.08),
}

# The Franka reaches comfortably to ~0.7 m; on the big boards the far rank is at
# the edge of that, so episodes only ever command pieces inside this radius.
MAX_PIECE_REACH = 0.68

_FRANKA_HOME_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.35,
    "panda_joint3": 0.0,
    "panda_joint4": -2.20,
    "panda_joint5": 0.0,
    "panda_joint6": 1.90,
    "panda_joint7": 0.785,
    "panda_finger_joint.*": 0.04,
}


def piece_usd_path(kind: str, scale_tag: str = PIECE_SCALE_TAG) -> str:
    """Path to a baked rigid-body piece asset."""
    path = GENERATED_ASSET_DIR / scale_tag / f"{kind}.usd"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing baked chess asset {path}. Run:\n"
            f"    python lab/scripts/prepare_chess_assets.py --scale {int(scale_tag[1:]) / 100}"
        )
    return str(path)


@configclass
class FrankaChessSceneCfg(InteractiveSceneCfg):
    """Table, Franka, chess board; the pieces are attached by :func:`configure_chess_scene`."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.18, 0.12), roughness=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z / 2.0)),
    )

    # Both are filled in by ChessPickEnvCfg.set_robot(): which arm, where its
    # end-effector prim lives and where its TCP sits are all robot-specific.
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.92, 0.92, 0.92), intensity=2600.0),
    )


def _piece_cfg(spec: PieceSpec, layout: BoardLayout, scale_tag: str, board_center: tuple[float, float]) -> RigidObjectCfg:
    """Rigid-body config for one piece, placed on its starting square."""
    x, y = layout.square_center(spec.file, spec.rank)
    half_yaw = spec.yaw / 2.0
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/pieces/{spec.color}_{spec.kind}_{spec.index}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=piece_usd_path(spec.kind, scale_tag),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.1, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PIECE_COLORS[spec.color], roughness=0.35),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(board_center[0] + x, board_center[1] + y, TABLE_TOP_Z + 0.002),
            # Isaac Lab 3.0 quaternions are (x, y, z, w).
            rot=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
        ),
    )


def capture_tray_center(layout: BoardLayout, board_center: tuple[float, float]) -> tuple[float, float]:
    """Tray centre in the environment frame, parked clear of the board's -y edge."""
    tray_half = (CAPTURE_TRAY_SHAPE[1] * CAPTURE_TRAY_PITCH) / 2.0
    return board_center[0], board_center[1] - (layout.half_width + CAPTURE_TRAY_GAP + tray_half)


def configure_chess_scene(
    scene: FrankaChessSceneCfg, layout: BoardLayout, scale_tag: str, board_center: tuple[float, float]
) -> None:
    """Attach the board, the capture tray and one rigid body per piece to an unbuilt scene."""
    for name in tuple(vars(scene)):
        if name in ("chessboard", "capture_tray") or name.startswith("piece_"):
            delattr(scene, name)

    board_x, board_y = layout.board_prim_offset
    half_yaw = layout.board_prim_yaw / 2.0
    scene.chessboard = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/chessboard",
        spawn=sim_utils.UsdFileCfg(
            usd_path=layout.board_usd_path, scale=(layout.board_scale, layout.board_scale, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(board_center[0] + board_x, board_center[1] + board_y, TABLE_TOP_Z + 0.001),
            rot=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
        ),
    )

    tray_x, tray_y = capture_tray_center(layout, board_center)
    scene.capture_tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/capture_tray",
        spawn=sim_utils.CuboidCfg(
            size=(CAPTURE_TRAY_SHAPE[0] * CAPTURE_TRAY_PITCH, CAPTURE_TRAY_SHAPE[1] * CAPTURE_TRAY_PITCH, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.42, 0.38), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(tray_x, tray_y, TABLE_TOP_Z + 0.0005)),
    )

    for spec in layout.pieces:
        setattr(scene, spec.name, _piece_cfg(spec, layout, scale_tag, board_center))


@configclass
class ActionsCfg:
    """Absolute task-space pose for the arm, binary open/close for the gripper.

    Both are filled in by :meth:`ChessPickEnvCfg.set_robot` -- the joint patterns,
    the IK body and the gripper's open/close targets all differ per arm.
    """

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm = MISSING


@configclass
class CommandsCfg:
    chess_move: mdp.ChessMoveCommandCfg = mdp.ChessMoveCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos, params={"gripper_joints": []})
        gripper_open = ObsTerm(func=mdp.gripper_open_fraction, params={"open_command": {}, "close_command": {}})
        chess_move = ObsTerm(func=mdp.generated_commands, params={"command_name": "chess_move"})
        target_piece = ObsTerm(func=mdp.commanded_piece_pose)
        piece_positions = ObsTerm(func=mdp.piece_positions, params={"piece_names": []})
        piece_orientations = ObsTerm(func=mdp.piece_orientations, params={"piece_names": []})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Per-subtask signals, in the layout Isaac Lab Mimic expects."""

        grasp = ObsTerm(func=mdp.piece_grasped, params={"open_command": {}, "close_command": {}})
        lift = ObsTerm(func=mdp.piece_lifted, params={"board_height": TABLE_TOP_Z})
        place = ObsTerm(func=mdp.piece_placed)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )
    reset_pieces = EventTerm(
        func=mdp.reset_pieces_on_board,
        mode="reset",
        params={"piece_names": [], "position_noise": 0.004, "yaw_noise": 0.35},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    piece_off_board = DoneTerm(func=mdp.any_piece_off_board, params={"piece_names": [], "minimum_height": 0.5})
    board_disturbed = DoneTerm(func=mdp.any_piece_toppled, params={"piece_names": []})
    success = DoneTerm(func=mdp.move_completed, params={"open_command": {}, "close_command": {}})


@configclass
class ChessPickEnvCfg(ManagerBasedRLEnvCfg):
    """An arm picking and placing chess pieces, driven by absolute task-space poses.

    Robot-agnostic: :meth:`set_robot` swaps in any entry of
    :data:`~robochess.tasks.manager_based.chess.robot_configs.CHESS_ROBOTS`.
    """

    robot_name: str = "franka"
    chess_scenario: str = "pieces"
    """Default to the compact one-of-each-kind board: it is the only layout every
    supported arm can reach, and it covers all six piece kinds."""
    piece_scale_tag: str = PIECE_SCALE_TAG
    board_scale: float | None = None
    """Square stretch factor; ``None`` picks the per-scenario default."""

    scene: FrankaChessSceneCfg = FrankaChessSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # A full pick-and-place interpolates for ~10.5 s and each leg may wait for the
        # arm to converge (up to ~10 s in total), so leave headroom before the time-out
        # cuts a good demo -- but not so much that failed attempts idle for a minute.
        self.episode_length_s = 24.0
        self.viewer.eye = (1.30, 1.10, 1.45)
        self.viewer.lookat = (0.20, 0.0, 0.80)
        self.set_robot(self.robot_name)
        self.set_chess_scenario(self.chess_scenario)

    """
    Operations.
    """

    def chess_robot(self) -> ChessRobotSpec:
        """Spec of the arm currently configured. A method, for the same reason as
        :meth:`chess_layout`."""
        return CHESS_ROBOTS[self.robot_name]

    def set_robot(self, robot_name: str) -> None:
        """Point every robot-dependent manager term at ``robot_name``.

        Which joints the IK drives, which prim the end-effector frame tracks, how far
        the TCP sits from it and what "gripper open" means are all per-arm, so they
        are read from the measured :class:`ChessRobotSpec` rather than hardcoded.
        """
        if robot_name not in CHESS_ROBOTS:
            raise ValueError(f"Unknown robot '{robot_name}'. Choose one of {CHESS_ROBOT_OPTIONS}.")
        self.robot_name = robot_name
        spec = CHESS_ROBOTS[robot_name]

        init_state = spec.articulation.init_state.replace(pos=spec.base_pos)
        if spec.home_joint_pos:
            init_state = init_state.replace(joint_pos=dict(spec.home_joint_pos))
        self.scene.robot = spec.articulation.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=init_state)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=spec.base_prim_path(),
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame"),
            target_frames=[
                # Index 0 is the TCP: every mdp term reads target_pos_w[:, 0].
                FrameTransformerCfg.FrameCfg(
                    prim_path=spec.ee_prim_path(), name="end_effector", offset=OffsetCfg(pos=spec.tcp_offset)
                ),
                FrameTransformerCfg.FrameCfg(prim_path=spec.ee_prim_path(), name="ee_body"),
            ],
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=list(spec.arm_joints),
            body_name=spec.ee_body,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(spec.gripper_joints),
            open_command_expr=dict(spec.open_command),
            close_command_expr=dict(spec.close_command),
        )

        # The reset event jogs only the arm joints, whose names are per-robot too.
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=list(spec.arm_joints))

        gripper_params = {"open_command": dict(spec.open_command), "close_command": dict(spec.close_command)}
        self.observations.policy.gripper_pos.params["gripper_joints"] = list(spec.gripper_joints)
        self.observations.policy.gripper_open.params.update(gripper_params)
        self.observations.subtask_terms.grasp.params.update(gripper_params)
        self.terminations.success.params.update(gripper_params)

        # Reach and mounting differ per arm, so the reachable-square filter has to be
        # recomputed -- but only once the board exists, since __post_init__ selects the
        # robot before it lays the board out.
        if hasattr(self.scene, "chessboard"):
            self.set_chess_scenario(self.chess_scenario)

    def chess_layout(self) -> BoardLayout:
        """Board layout currently configured on this env.

        Deliberately a method, not a property: Isaac Lab's ``@configclass`` walks
        every attribute it can reach to resolve string-valued callables, which
        evaluates properties and then tries to write into whatever comes back --
        and :class:`BoardLayout` is frozen.
        """
        return make_layout(self.chess_scenario, self._resolved_board_scale(self.chess_scenario))

    def _resolved_board_scale(self, scenario: str) -> float:
        return self.board_scale if self.board_scale is not None else DEFAULT_BOARD_SCALE[scenario]

    def set_chess_scenario(self, scenario: str, board_scale: float | None = None) -> None:
        """Rebuild the scene and every layout-dependent manager term for ``scenario``."""
        self.chess_scenario = scenario
        if board_scale is not None:
            self.board_scale = board_scale
        layout = make_layout(scenario, self._resolved_board_scale(scenario))
        configure_chess_scene(self.scene, layout, self.piece_scale_tag, self.board_center())

        piece_names = [spec.name for spec in layout.pieces]
        piece_kinds = [spec.kind for spec in layout.pieces]

        self.observations.policy.piece_positions.params["piece_names"] = piece_names
        self.observations.policy.piece_orientations.params["piece_names"] = piece_names
        self.events.reset_pieces.params["piece_names"] = piece_names
        self.terminations.piece_off_board.params["piece_names"] = piece_names
        self.terminations.board_disturbed.params["piece_names"] = piece_names

        self.commands.chess_move.piece_names = piece_names
        self.commands.chess_move.piece_kinds = piece_kinds
        self.commands.chess_move.movable_piece_indices = self._reachable_pieces(layout)
        self.commands.chess_move.target_positions = self._target_positions(layout)

    def start_positions(self, layout: BoardLayout | None = None) -> list[tuple[float, float, float]]:
        """Environment-frame start position of every piece, in ``layout.pieces`` order."""
        layout = layout or self.chess_layout()
        return [self.board_square_pos(*layout.square_center(spec.file, spec.rank)) for spec in layout.pieces]

    def board_center(self) -> tuple[float, float]:
        """Board centre in the environment frame, ``board_distance`` in front of the base."""
        spec = self.chess_robot()
        return spec.base_pos[0] + spec.board_distance, spec.base_pos[1]

    def board_square_pos(self, board_x: float, board_y: float) -> tuple[float, float, float]:
        """Board-frame square centre to an environment-frame resting position."""
        center = self.board_center()
        return center[0] + board_x, center[1] + board_y, TABLE_TOP_Z

    def capture_tray_positions(self, layout: BoardLayout) -> list[tuple[float, float, float]]:
        """Environment-frame slots of the off-board capture tray."""
        columns, rows = CAPTURE_TRAY_SHAPE
        center_x, center_y = capture_tray_center(layout, self.board_center())
        return [
            (
                center_x + (col - (columns - 1) / 2.0) * CAPTURE_TRAY_PITCH,
                center_y + (row - (rows - 1) / 2.0) * CAPTURE_TRAY_PITCH,
                TABLE_TOP_Z,
            )
            for col in range(columns)
            for row in range(rows)
        ]

    def _reachable_pieces(self, layout: BoardLayout) -> list[int]:
        """Piece indices whose start square the Franka can comfortably reach."""
        reachable = [
            index
            for index, position in enumerate(self.start_positions(layout))
            if self._distance_from_base(position) <= self.chess_robot().reach
        ]
        if not reachable:
            raise ValueError(f"No chess piece in scenario '{layout.scenario}' is within the Franka's reach.")
        return reachable

    def _target_positions(self, layout: BoardLayout) -> list[tuple[float, float, float]]:
        """Destinations that are both reachable and safely on the table.

        On a wide board the tray gets pushed far enough out that some of its slots
        hang over the table edge, where a released piece just falls off.
        """
        candidates = [
            self.board_square_pos(*layout.square_center(file, rank)) for file, rank in layout.free_squares()
        ] + self.capture_tray_positions(layout)
        targets = [
            pos for pos in candidates
            if self._distance_from_base(pos) <= self.chess_robot().reach and self._on_table(pos)
        ]
        if not targets:
            raise ValueError(f"Scenario '{layout.scenario}' has no reachable destination for a piece.")
        return targets

    def _distance_from_base(self, position: tuple[float, float, float]) -> float:
        base = self.chess_robot().base_pos
        return ((position[0] - base[0]) ** 2 + (position[1] - base[1]) ** 2) ** 0.5

    @staticmethod
    def _on_table(position: tuple[float, float, float]) -> bool:
        return (
            abs(position[0] - TABLE_CENTER[0]) <= TABLE_SIZE[0] / 2.0 - TABLE_EDGE_MARGIN
            and abs(position[1] - TABLE_CENTER[1]) <= TABLE_SIZE[1] / 2.0 - TABLE_EDGE_MARGIN
        )


@configclass
class FrankaChessEnvCfg(ChessPickEnvCfg):
    """Backwards-compatible alias: the chess pick task with the Franka on 4x4 minichess."""

    robot_name: str = "franka"
    chess_scenario: str = "4x4"


@configclass
class FrankaChessRelEnvCfg(FrankaChessEnvCfg):
    """Same task, but the arm takes relative task-space deltas (teleop / policy friendly)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
        )
