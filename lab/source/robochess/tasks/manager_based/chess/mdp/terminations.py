"""Termination terms for the RoboChess manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from .observations import piece_grasped, piece_placed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def move_completed(
    env: ManagerBasedRLEnv,
    open_command: dict[str, float],
    close_command: dict[str, float],
    command_name: str = "chess_move",
    xy_threshold: float = 0.02,
    height_threshold: float = 0.01,
) -> torch.Tensor:
    """Success: the commanded piece stands upright and settled on its target square.

    The piece must also be out of the gripper. Without that check the episode ends
    the instant the piece touches down -- while it is still held -- and the demo is
    truncated before the release, which makes the recording unusable for imitation.
    """
    placed = piece_placed(
        env, command_name=command_name, xy_threshold=xy_threshold, height_threshold=height_threshold
    ).squeeze(-1)
    held = piece_grasped(env, open_command, close_command, command_name=command_name)
    return placed & ~held.squeeze(-1)


def any_piece_off_board(
    env: ManagerBasedRLEnv,
    piece_names: list[str],
    minimum_height: float = -0.05,
) -> torch.Tensor:
    """Failure: a piece fell off the table (height measured relative to the env origin)."""
    origins_z = env.scene.env_origins[:, 2]
    heights = torch.stack([env.scene[name].data.root_pos_w.torch[:, 2] - origins_z for name in piece_names], dim=1)
    return torch.any(heights < minimum_height, dim=1)


def any_piece_toppled(
    env: ManagerBasedRLEnv,
    piece_names: list[str],
    command_name: str = "chess_move",
    max_tilt_deg: float = 30.0,
) -> torch.Tensor:
    """Failure: a piece that is not being carried is no longer standing up.

    A demonstration that completes the commanded move but knocks a neighbour over on
    the way is worse than useless for imitation -- it teaches the policy to barge
    through the board. Ending those episodes keeps them out of the dataset.
    """
    quats = torch.stack([env.scene[name].data.root_quat_w.torch for name in piece_names], dim=1)
    num_envs, num_pieces = quats.shape[:2]

    up_axis = torch.tensor([0.0, 0.0, 1.0], device=quats.device).expand(num_envs * num_pieces, 3)
    piece_up = math_utils.quat_apply(quats.reshape(-1, 4), up_axis).reshape(num_envs, num_pieces, 3)
    tilted = piece_up[:, :, 2] < torch.cos(torch.deg2rad(torch.tensor(max_tilt_deg, device=quats.device)))

    # The carried piece is allowed to be tilted -- the grasp itself is tilted.
    carried = env.command_manager.get_term(command_name).piece_index
    tilted[torch.arange(num_envs, device=quats.device), carried] = False
    return torch.any(tilted, dim=1)
