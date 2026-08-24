"""Gym registration for the RoboChess manager-based chess tasks."""

import gymnasium as gym

gym.register(
    id="RoboChess-Visual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.chess_env_cfg:RoboChessVisualEnvCfg"},
)

##
# Franka Panda chess pick-and-place.
##

gym.register(
    id="RoboChess-Franka-Chess-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.franka_chess_env_cfg:FrankaChessEnvCfg"},
)

gym.register(
    id="RoboChess-Franka-Chess-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.franka_chess_env_cfg:FrankaChessRelEnvCfg"},
)

# Robot-agnostic form of the same task: pick the arm with cfg.set_robot(name), or
# with --robot on the generation and render scripts.
gym.register(
    id="RoboChess-Chess-Pick-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.franka_chess_env_cfg:ChessPickEnvCfg"},
)
