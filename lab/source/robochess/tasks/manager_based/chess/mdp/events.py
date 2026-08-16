"""Event terms for the RoboChess manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv


def reset_pieces_on_board(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    piece_names: list[str],
    position_noise: float = 0.004,
    yaw_noise: float = 0.35,
) -> None:
    """Return every piece to its starting square with a little placement noise.

    Real boards are never set up perfectly, and a demo dataset in which the piece
    is always at exactly the same pose teaches a policy nothing about closing the
    loop on perception. The noise stays well inside a square so the setup remains
    a legal position.
    """
    for name in piece_names:
        asset: RigidObject = env.scene[name]
        default_pose = asset.data.default_root_pose.torch[env_ids].clone()

        offsets = torch.zeros((len(env_ids), 3), device=asset.device)
        offsets[:, :2] = math_utils.sample_uniform(-position_noise, position_noise, (len(env_ids), 2), asset.device)
        positions = default_pose[:, 0:3] + env.scene.env_origins[env_ids] + offsets

        yaw = math_utils.sample_uniform(-yaw_noise, yaw_noise, (len(env_ids),), asset.device)
        zeros = torch.zeros_like(yaw)
        orientations = math_utils.quat_mul(default_pose[:, 3:7], math_utils.quat_from_euler_xyz(zeros, zeros, yaw))

        asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids), 6), device=asset.device), env_ids=env_ids
        )
