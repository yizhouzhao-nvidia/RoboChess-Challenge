"""Robot configurations available to the RoboChess visual and picking tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import UR10_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from robochess.tasks.manager_based.flexiv_rizon.flexiv_rizon_robots import FLEXIV_RIZON_4S_CFG

ASSET_DIR = Path(__file__).resolve().parents[6] / "assets"

# Isaac Sim 6.0 replaced Robots/FrankaEmika/panda_instanceable.usd with
# franka_panda.usda, but isaaclab_assets.robots.franka still points at the old
# path, so loading FRANKA_PANDA_CFG as shipped raises FileNotFoundError. Body and
# joint names are unchanged, so only the path needs patching.
FRANKA_PANDA_USD_PATH = os.environ.get(
    "FRANKA_PANDA_USD_PATH", f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/franka_panda.usda"
)

FRANKA_PANDA_CHESS_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_PANDA_CHESS_CFG.spawn.usd_path = FRANKA_PANDA_USD_PATH

# FRANKA_PANDA_HIGH_PD_CFG relies on ``disable_gravity`` to keep differential-IK
# tracking tight, but ``modify_rigid_body_properties`` stops descending once it
# finds a rigid body, and this asset nests every link inside its parent -- so only
# panda_link0 ends up with the flag and the other ten links sag. That leaves a
# 30-50 mm steady-state Cartesian error, which is enough to miss a chess piece
# entirely. Stiffer position gains hold the arm against gravity instead, which is
# also closer to the real robot's gravity-compensated controller.
FRANKA_PANDA_CHESS_CFG.actuators["panda_shoulder"].stiffness = 9000.0
FRANKA_PANDA_CHESS_CFG.actuators["panda_shoulder"].damping = 600.0
FRANKA_PANDA_CHESS_CFG.actuators["panda_forearm"].stiffness = 4500.0
FRANKA_PANDA_CHESS_CFG.actuators["panda_forearm"].damping = 300.0

# The 6.0 asset also nests each link inside its parent under a "Geometry" scope
# instead of the flat layout of panda_instanceable.usd. Articulation joint/body
# *names* are unaffected, but anything addressing a body by prim path -- notably
# FrameTransformerCfg -- needs the full chain.
_FRANKA_LINK_CHAIN = (
    "Geometry",
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
)
_FRANKA_FINGERS = ("panda_leftfinger", "panda_rightfinger")


def franka_body_prim_path(body_name: str, root: str = "{ENV_REGEX_NS}/Robot") -> str:
    """Prim path of a Franka body inside :data:`FRANKA_PANDA_USD_PATH`."""
    if body_name in _FRANKA_FINGERS:
        return "/".join((root, *_FRANKA_LINK_CHAIN, body_name))
    try:
        depth = _FRANKA_LINK_CHAIN.index(body_name)
    except ValueError as error:
        raise ValueError(f"Unknown Franka body '{body_name}'. Known: {_FRANKA_LINK_CHAIN + _FRANKA_FINGERS}") from error
    return "/".join((root, *_FRANKA_LINK_CHAIN[: depth + 1]))


SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "so101" / "TheRobotStudio" / "so101_new_calib" / "so101_new_calib_physx.usd"),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim=10.0,
            stiffness=200.0,
            damping=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"], effort_limit_sim=5.0, stiffness=100.0, damping=5.0
        ),
    },
)

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "piper" / "piper_camera.usd"),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={f"joint{i}": 0.0 for i in range(1, 9)}
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[f"joint{i}" for i in range(1, 7)],
            effort_limit_sim=50.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint7", "joint8"], effort_limit_sim=20.0, stiffness=500.0, damping=20.0
        ),
    },
)

REBOT_USD_PATH = os.environ.get(
    "REBOT_USD_PATH",
    "https://raw.githubusercontent.com/Seeed-Projects/reBot-Isaacsim/refs/heads/main/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda",
)
REBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=REBOT_USD_PATH, activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint_left": 0.02,
            "joint_right": 0.02,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"], effort_limit_sim=36.0, stiffness=500.0, damping=50.0
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint_left", "joint_right"], effort_limit_sim=100.0, stiffness=100.0, damping=4.0
        ),
    },
)

YAM_USD_PATH = os.environ.get(
    "YAM_USD_PATH",
    "https://github.com/ARISE-Initiative/yamlab/raw/refs/heads/main/yamlab/robot/yam/arm/yam.usd",
)
YAM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=YAM_USD_PATH, activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "left_finger": -0.04695,
            "right_finger": -0.04695,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"], effort_limit_sim=20.0, stiffness=120.0, damping=12.0
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger", "right_finger"], effort_limit_sim=20.0, stiffness=200.0, damping=2.0
        ),
    },
)

ROBOT_OPTIONS = ("so101", "piper", "ur10", "flexiv_rizon", "rebot", "yam")

_ROBOT_CONFIGS = {
    "so101": (SO101_CFG, (-0.10, 0.0, 0.782)),
    "piper": (PIPER_CFG, (-0.25, 0.0, 0.772)),
    "ur10": (UR10_CFG, (-0.42, 0.0, 0.77)),
    "flexiv_rizon": (FLEXIV_RIZON_4S_CFG, (-0.42, 0.0, 0.77)),
    "rebot": (REBOT_CFG, (-0.20, 0.0, 0.77)),
    "yam": (YAM_CFG, (-0.20, 0.0, 0.77)),
}


def make_robot_cfg(robot_name: str) -> ArticulationCfg:
    """Return the selected robot at its single-arm RoboChess mounting location."""
    try:
        robot_cfg, position = _ROBOT_CONFIGS[robot_name]
    except KeyError as error:
        raise ValueError(f"Unsupported robot {robot_name}. Choose one of {ROBOT_OPTIONS}.") from error
    return robot_cfg.replace(
        prim_path="{ENV_REGEX_NS}/robot",
        init_state=robot_cfg.init_state.replace(pos=position),
    )


##
# Chess picking: what an arm needs beyond its ArticulationCfg.
##


@dataclass
class ChessRobotSpec:
    """Everything the chess picking task needs that a USD path cannot tell it.

    Deliberately a plain dataclass rather than a ``@configclass``: it is built on
    demand from a method, never stored on an env config, so Isaac Lab's config
    walker never sees it.

    Every geometric field here is *measured*, not guessed -- run
    ``lab/scripts/probe_robot.py --robot <key>`` and read them off.
    """

    key: str
    articulation: ArticulationCfg
    base_pos: tuple[float, float, float]
    """Where the arm is bolted to the table, in the environment frame."""

    arm_joints: list[str]
    gripper_joints: list[str]

    ee_body: str
    """Body the differential IK drives; :attr:`approach_axis` is given in its frame."""
    base_relative_prim_path: str
    """Path of the arm's base link prim relative to the spawned robot root.

    Only used as the ``FrameTransformerCfg`` source frame, which must resolve to a
    rigid body.
    """

    ee_relative_prim_path: str
    """Path of :attr:`ee_body`'s prim relative to the spawned robot root.

    Needed because ``FrameTransformerCfg`` addresses bodies by prim path, and USD
    assets differ wildly: some lay every link out flat under the root, others nest
    each link inside its parent.
    """

    tcp_offset: tuple[float, float, float]
    """Translation from :attr:`ee_body` to the point between the fingertips."""

    approach_axis: tuple[float, float, float]
    """Unit vector, in :attr:`ee_body` coordinates, pointing out of the gripper."""

    closing_axis: tuple[float, float, float]
    """Unit vector, in :attr:`ee_body` coordinates, along which the fingers travel."""

    open_command: dict[str, float]
    close_command: dict[str, float]
    max_opening: float
    """Total finger separation when fully open [m]; must exceed the grasp span."""

    reach: float
    """Comfortable planar reach from the base [m]; bounds which squares are used."""

    board_distance: float = 0.45
    """How far in front of the base the board centre is placed [m].

    Per-arm because these robots differ by a factor of two in reach: a board that is
    comfortable for a Panda is entirely outside a reBot's workspace.
    """

    home_joint_pos: dict[str, float] = field(default_factory=dict)
    """Posture the arm resets to.

    Not cosmetic: several of these assets have joint limits that make the all-zeros
    pose a fully extended arm lying through the table, which knocks the board over
    before the first control step.
    """

    finger_bodies: list[str] = field(default_factory=list)
    """Optional finger bodies, used by the probe and by debug visualisation."""

    def ee_prim_path(self, root: str = "{ENV_REGEX_NS}/Robot") -> str:
        return f"{root}/{self.ee_relative_prim_path}"

    def base_prim_path(self, root: str = "{ENV_REGEX_NS}/Robot") -> str:
        return f"{root}/{self.base_relative_prim_path}"

    def graspgen_to_ee(self, graspgen_depth: float) -> np.ndarray:
        """4x4 transform taking a GraspGen grasp pose to this arm's ``ee_body`` pose.

        GraspGen's convention is approach ``+Z``, fingers closing along ``+X``, origin
        at the gripper base with the TCP at ``+Z * depth``. Real arms disagree on both
        axes -- the Franka closes along its hand's Y, the reBot reaches along its
        ``gripper_end``'s X -- so the mapping is built from the measured axes rather
        than assumed to be a rotation about one of them.

        The rotation carries GraspGen's axes onto this gripper's, and the translation
        places ``ee_body`` so that the two TCPs coincide.
        """
        approach = np.asarray(self.approach_axis, dtype=float)
        closing = np.asarray(self.closing_axis, dtype=float)
        approach /= np.linalg.norm(approach)
        closing /= np.linalg.norm(closing)
        # Rows, not columns: this maps a vector from ee coordinates into GraspGen's,
        # so it must send this gripper's closing axis to GraspGen's +X and its
        # approach axis to GraspGen's +Z. Building it column-wise gives the transpose,
        # which is indistinguishable for an arm whose approach is already +Z (the
        # error is then just a harmless +-90 degrees about a symmetric gripper's own
        # axis) and sends one whose approach is +X off sideways by 90 degrees.
        rotation = np.array([closing, np.cross(approach, closing), approach])

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array([0.0, 0.0, graspgen_depth]) - rotation @ np.asarray(self.tcp_offset, dtype=float)
        return transform


_FRANKA_CHESS_HOME = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.35,
    "panda_joint3": 0.0,
    "panda_joint4": -2.20,
    "panda_joint5": 0.0,
    "panda_joint6": 1.90,
    "panda_joint7": 0.785,
    "panda_finger_joint.*": 0.04,
}

# Isaac Sim's piper_v2.usd is a full *scene* (ground plane, lights, render settings),
# not an articulation, so the picking task points at the robot layer that scene
# references. Loaded straight from the URL -- no local copy.
PIPER_CHESS_USD_PATH = os.environ.get(
    "PIPER_CHESS_USD_PATH",
    "https://raw.githubusercontent.com/agilexrobotics/piper_isaac_sim/refs/heads/master"
    "/piper_description/urdf/piper_description_v100_realsense_camera_v2"
    "/piper_description_v100_realsense_camera_v2.usd",
)


def _chess_tuned(cfg: ArticulationCfg, arm_expr: str, stiffness: float, damping: float) -> ArticulationCfg:
    """Stiffen an arm's position gains for accurate differential-IK tracking.

    Chess pieces are gripped on a 19-43 mm shaft, so a Cartesian steady-state error
    of even a centimetre misses. Stock gains on these arms leave 30-50 mm of gravity
    sag; the real robots all run gravity-compensated controllers, so stiff position
    control is the closer analogue.
    """
    tuned = cfg.copy()
    for actuator in tuned.actuators.values():
        if any(expr.startswith(arm_expr[:5]) or expr == arm_expr for expr in actuator.joint_names_expr):
            actuator.stiffness = stiffness
            actuator.damping = damping
    return tuned


PIPER_CHESS_CFG = PIPER_CFG.copy()
PIPER_CHESS_CFG.spawn.usd_path = PIPER_CHESS_USD_PATH
# Same story as the Franka: these arms sag under gravity because the *effort*
# ceiling binds long before the position gain does -- a constant ~50 mm Cartesian
# offset at every target, independent of distance, which is the giveaway. Real
# arms gravity-compensate, so raising the sim ceiling is the faithful analogue.
PIPER_CHESS_CFG.actuators["arm"].stiffness = 4000.0
PIPER_CHESS_CFG.actuators["arm"].damping = 300.0
PIPER_CHESS_CFG.actuators["arm"].effort_limit_sim = 400.0
PIPER_CHESS_CFG.actuators["gripper"].stiffness = 2000.0
PIPER_CHESS_CFG.actuators["gripper"].damping = 100.0

REBOT_CHESS_CFG = REBOT_CFG.copy()
REBOT_CHESS_CFG.actuators["arm"].stiffness = 4000.0
REBOT_CHESS_CFG.actuators["arm"].damping = 300.0
REBOT_CHESS_CFG.actuators["arm"].effort_limit_sim = 300.0
REBOT_CHESS_CFG.actuators["gripper"].stiffness = 2000.0
REBOT_CHESS_CFG.actuators["gripper"].damping = 100.0

YAM_CHESS_CFG = YAM_CFG.copy()
YAM_CHESS_CFG.actuators["arm"].stiffness = 3000.0
YAM_CHESS_CFG.actuators["arm"].damping = 250.0
YAM_CHESS_CFG.actuators["arm"].effort_limit_sim = 300.0
YAM_CHESS_CFG.actuators["gripper"].stiffness = 2000.0
YAM_CHESS_CFG.actuators["gripper"].damping = 100.0
# The two YAM finger meshes overlap in their rest pose, so with self-collisions on
# PhysX fights the gripper drive and the fingers never leave ~1 mm of travel however
# hard they are pushed. Disabling self-collisions (as the upstream YAM config does)
# and raising the effort ceiling makes the joints track their command exactly.
YAM_CHESS_CFG.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)
YAM_CHESS_CFG.actuators["gripper"].effort_limit_sim = 500.0
YAM_CHESS_CFG.init_state = ArticulationCfg.InitialStateCfg(
    joint_pos={**{f"joint{i}": 0.0 for i in range(1, 7)}, "left_finger": 0.0, "right_finger": 0.0}
)


CHESS_ROBOTS: dict[str, ChessRobotSpec] = {
    "franka": ChessRobotSpec(
        key="franka",
        articulation=FRANKA_PANDA_CHESS_CFG.replace(
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=_FRANKA_CHESS_HOME)
        ),
        base_pos=(-0.23, 0.0, 0.77),
        arm_joints=["panda_joint.*"],
        gripper_joints=["panda_finger_joint.*"],
        base_relative_prim_path="Geometry/panda_link0",
        ee_body="panda_hand",
        ee_relative_prim_path="/".join(_FRANKA_LINK_CHAIN),
        tcp_offset=(0.0, 0.0, 0.1034),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(0.0, 1.0, 0.0),
        open_command={"panda_finger_.*": 0.04},
        close_command={"panda_finger_.*": 0.0},
        max_opening=0.08,
        reach=0.68,
        board_distance=0.45,
        home_joint_pos=dict(_FRANKA_CHESS_HOME),
        finger_bodies=list(_FRANKA_FINGERS),
    ),
    "piper": ChessRobotSpec(
        key="piper",
        articulation=PIPER_CHESS_CFG,
        base_pos=(-0.20, 0.0, 0.77),
        arm_joints=["joint[1-6]"],
        gripper_joints=["joint7", "joint8"],
        base_relative_prim_path="arm_base",
        ee_body="link6",
        ee_relative_prim_path="link6",
        tcp_offset=(0.0, 0.0, 0.125),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(1.0, 0.0, 0.0),
        open_command={"joint7": 0.035, "joint8": -0.035},
        close_command={"joint7": 0.0, "joint8": 0.0},
        max_opening=0.07,
        reach=0.42,
        board_distance=0.30,
        home_joint_pos={"joint1": 0.0, "joint2": 1.25, "joint3": -1.55, "joint4": 0.0, "joint5": 0.95, "joint6": 0.0},
        finger_bodies=["link7", "link8"],
    ),
    "rebot": ChessRobotSpec(
        key="rebot",
        articulation=REBOT_CHESS_CFG,
        base_pos=(-0.20, 0.0, 0.77),
        arm_joints=["joint[1-6]"],
        gripper_joints=["joint_left", "joint_right"],
        base_relative_prim_path="Geometry/base_link",
        ee_body="gripper_end",
        ee_relative_prim_path="Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end",
        tcp_offset=(-0.015, 0.0, 0.0),
        approach_axis=(1.0, 0.0, 0.0),
        closing_axis=(0.0, 1.0, 0.0),
        open_command={"joint_left": 0.045, "joint_right": 0.045},
        close_command={"joint_left": 0.0, "joint_right": 0.0},
        max_opening=0.09,
        reach=0.44,
        board_distance=0.30,
        home_joint_pos={"joint1": 0.0, "joint2": -1.25, "joint3": -1.55, "joint4": 0.0, "joint5": -0.75, "joint6": 0.0},
        finger_bodies=["gripper_left", "gripper_right"],
    ),
    "yam": ChessRobotSpec(
        key="yam",
        articulation=YAM_CHESS_CFG,
        base_pos=(-0.20, 0.0, 0.77),
        arm_joints=["joint[1-6]"],
        gripper_joints=["left_finger", "right_finger"],
        base_relative_prim_path="arm/arm",
        ee_body="link_6",
        ee_relative_prim_path="arm/link_6",
        # Bracketed empirically: 0.09 and 0.145 both give 0 successes in 16 attempts,
        # 0.13 gives 12%. The probe can locate the finger *bodies* but not the pad
        # face they grip with, so the last centimetre has to be found by trying.
        tcp_offset=(0.0, 0.0, 0.13),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(1.0, 0.0, 0.0),
        # Measured, and the reverse of what the upstream config's comment implies:
        # joint 0 holds the fingers 90 mm apart, -0.0475 brings them together.
        open_command={"left_finger": 0.0, "right_finger": 0.0},
        close_command={"left_finger": -0.0475, "right_finger": -0.0475},
        max_opening=0.090,
        reach=0.42,
        board_distance=0.30,
        home_joint_pos={"joint1": 0.0, "joint2": 1.25, "joint3": 1.25, "joint4": 0.0, "joint5": 0.85, "joint6": 0.0},
        finger_bodies=["left_finger", "right_finger"],
    ),
}

CHESS_ROBOT_OPTIONS = tuple(CHESS_ROBOTS)
