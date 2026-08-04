"""Robot configurations available to the RoboChess visual task."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import UR10_CFG

from robochess.tasks.manager_based.flexiv_rizon.flexiv_rizon_robots import FLEXIV_RIZON_4S_CFG

ASSET_DIR = Path(__file__).resolve().parents[6] / "assets"

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
