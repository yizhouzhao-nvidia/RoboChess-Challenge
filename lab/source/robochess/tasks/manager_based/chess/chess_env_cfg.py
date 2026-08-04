"""Minimal manager-based Isaac Lab scene for visually inspecting RoboChess."""

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from .robot_configs import make_robot_cfg

_TABLE_TOP_Z = 0.77
_SQUARE_SIZE = 0.04
_BOARD_CENTER = (0.20, 0.0)
_ASSET_DIR = Path(__file__).resolve().parents[6] / "assets"
_CHESS_ASSET_DIR = _ASSET_DIR / "chess"


def _visual_cuboid(size: tuple[float, float, float], color: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.65),
    )


def _visual_piece(color: tuple[float, float, float], height: float) -> sim_utils.CylinderCfg:
    return sim_utils.CylinderCfg(
        radius=0.027,
        height=height,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.35),
    )


SUPPORTED_CHESS_SCENARIOS = ("1d", "3x3", "4x4", "8x8")


def _piece_layout(scenario: str) -> tuple[str, tuple[float, float], float, list[tuple[str, str, int, float, float, tuple[float, float, float, float]]]]:
    """Return board asset, board origin, board scale, and legacy-compatible pieces."""
    base_x, base_y = _BOARD_CENTER
    size = 0.04
    identity = (0.0, 0.0, 0.0, 1.0)
    white_knight = (0.0, 0.0, 0.7071, 0.7071)
    black_knight = (0.0, 0.0, -0.7071, 0.7071)

    if scenario == "1d":
        pieces = [
            ("white", "king", 0, base_x - 4 * size, base_y, identity),
            ("white", "knight", 0, base_x - 3 * size, base_y, white_knight),
            ("white", "rook", 0, base_x - 2 * size, base_y, identity),
            ("black", "rook", 0, base_x + size, base_y, identity),
            ("black", "knight", 0, base_x + 2 * size, base_y, black_knight),
            ("black", "king", 0, base_x + 3 * size, base_y, identity),
        ]
        return "board_1x6.usdc", (base_x - 4 * size, base_y), 1.0, pieces

    if scenario == "3x3":
        pieces = []
        for index, y in enumerate((-size, 0.0, size)):
            pieces.append(("white", "pawn", index, base_x - size, base_y + y, identity))
            pieces.append(("black", "pawn", index, base_x + size, base_y - y, identity))
        return "board_3x3.usdc", (base_x, base_y), 2 / 3, pieces

    if scenario == "4x4":
        pieces = []
        back_rank = ("king", "knight", "knight", "rook")
        for color, rank, knight_rot in (("white", -1.5, white_knight), ("black", 1.5, black_knight)):
            counts: dict[str, int] = {}
            for file_index, piece_name in enumerate(back_rank):
                index = counts.get(piece_name, 0)
                counts[piece_name] = index + 1
                rotation = knight_rot if piece_name == "knight" else identity
                pieces.append((color, piece_name, index, base_x + rank * size, base_y + (file_index - 1.5) * size, rotation))
            pawn_rank = -0.5 if color == "white" else 0.5
            for file_index in range(4):
                pieces.append((color, "pawn", file_index, base_x + pawn_rank * size, base_y + (file_index - 1.5) * size, identity))
        return "board_4x4.usdc", (base_x, base_y), 2 / 3, pieces

    if scenario == "8x8":
        pieces = []
        back_rank = ("rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook")
        for color, rank, pawn_rank, knight_rot in (
            ("white", -3.5, -2.5, white_knight),
            ("black", 3.5, 2.5, black_knight),
        ):
            counts: dict[str, int] = {}
            for file_index, piece_name in enumerate(back_rank):
                index = counts.get(piece_name, 0)
                counts[piece_name] = index + 1
                rotation = knight_rot if piece_name == "knight" else identity
                pieces.append((color, piece_name, index, base_x + rank * size, base_y + (file_index - 3.5) * size, rotation))
            for file_index in range(8):
                pieces.append((color, "pawn", file_index, base_x + pawn_rank * size, base_y + (file_index - 3.5) * size, identity))
        return "board_8x8.usdc", (base_x, base_y), 2 / 3, pieces

    raise ValueError(f"Unsupported chess scenario: {scenario}. Choose one of {SUPPORTED_CHESS_SCENARIOS}.")


@configclass
class RoboChessSceneCfg(InteractiveSceneCfg):
    """One UR10 and a chess scene configured before Isaac Lab creates the stage."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=_visual_cuboid((1.20, 0.90, 0.10), (0.22, 0.16, 0.10)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.15, 0.0, 0.72)),
    )
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )


def configure_chess_scene(scene: RoboChessSceneCfg, scenario: str) -> None:
    """Attach the selected board and real chess-piece assets to an unbuilt scene."""
    board_file, board_position, board_scale, pieces = _piece_layout(scenario)
    for name in tuple(vars(scene)):
        if name == "chessboard" or name.startswith("piece_"):
            delattr(scene, name)

    scene.chessboard = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/chessboard",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_CHESS_ASSET_DIR / "board" / board_file),
            scale=(board_scale, board_scale, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=board_position + (_TABLE_TOP_Z + 0.010,)),
    )
    for color, piece_name, index, x, y, rotation in pieces:
        asset_name = f"piece_{color}_{piece_name}_{index}"
        scene.__setattr__(
            asset_name,
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/pieces/{color}_{piece_name}_{index}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(_CHESS_ASSET_DIR / f"{piece_name}.usdc"),
                    scale=(2 / 3, 2 / 3, 2 / 3),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.92, 0.92, 0.88) if color == "white" else (0.04, 0.04, 0.05),
                        roughness=0.4,
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, _TABLE_TOP_Z + 0.012), rot=rotation),
            ),
        )


@configclass
class ActionsCfg:
    arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RoboChessVisualEnvCfg(ManagerBasedRLEnvCfg):
    """No-teleop, zero-action configuration for inspecting the chess scene."""

    robot_name: str = "ur10"
    chess_scenario: str = "4x4"
    scene: RoboChessSceneCfg = RoboChessSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 120.0
        self.viewer.eye = (1.65, 1.25, 1.35)
        self.viewer.lookat = (0.10, 0.0, 0.72)
        self.set_robot(self.robot_name)
        self.set_chess_scenario(self.chess_scenario)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.0, use_default_offset=True
        )

    def set_robot(self, robot_name: str) -> None:
        self.robot_name = robot_name
        self.scene.robot = make_robot_cfg(robot_name)

    def set_chess_scenario(self, scenario: str) -> None:
        self.chess_scenario = scenario
        configure_chess_scene(self.scene, scenario)
