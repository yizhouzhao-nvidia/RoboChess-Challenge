import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# 8 levels up from this file's directory reaches the project root
ASSET_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../../../../../..", "assets"))

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSET_DIR, "so101/TheRobotStudio/so101_new_calib/so101_new_calib.usd"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan": -0.0,
            "shoulder_lift": 0.0075,
            "elbow_flex": 0.0005,
            "wrist_flex": 0.0001,
            "wrist_roll": 0.0,
            "gripper": 0.0012,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            effort_limit_sim=10.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
)
