"""Gym registration for the single-arm RoboChess visual task."""

import gymnasium as gym


gym.register(
    id="RoboChess-UR10-Visual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.chess_env_cfg:RoboChessVisualEnvCfg"},
)
