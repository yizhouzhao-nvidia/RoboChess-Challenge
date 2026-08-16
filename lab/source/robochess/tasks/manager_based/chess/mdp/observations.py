"""Observation terms for the RoboChess manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector (TCP) position relative to the environment origin."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w.torch[:, 0, :] - env.scene.env_origins


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector (TCP) orientation as an (x, y, z, w) quaternion."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w.torch[:, 0, :]


def gripper_pos(
    env: ManagerBasedRLEnv,
    gripper_joints: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Raw finger joint positions."""
    robot: Articulation = env.scene[robot_cfg.name]
    finger_ids, _ = robot.find_joints(gripper_joints)
    return robot.data.joint_pos.torch[:, finger_ids]


def gripper_open_fraction(
    env: ManagerBasedRLEnv,
    open_command: dict[str, float],
    close_command: dict[str, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """How far open the gripper is, as 0 (shut) to 1 (wide), averaged over the fingers.

    Normalising against each joint's own open and closed targets is what makes this
    portable. Raw joint values cannot be compared across arms: the Franka opens at
    +0.04 and shuts at 0, the Piper's two fingers run in opposite directions, and
    YAM's 0 is *wide open* with -0.0475 shut. Fractions hide all of that.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    fractions = []
    for joint_expr, open_value in open_command.items():
        ids, _ = robot.find_joints(joint_expr)
        closed_value = close_command[joint_expr]
        span = open_value - closed_value
        if abs(span) < 1e-9:
            continue
        fractions.append((robot.data.joint_pos.torch[:, ids] - closed_value) / span)
    return torch.cat(fractions, dim=-1).mean(dim=-1, keepdim=True)


def piece_positions(env: ManagerBasedRLEnv, piece_names: list[str]) -> torch.Tensor:
    """Positions of every chess piece relative to the environment origin. Shape is (N, 3 * pieces)."""
    origins = env.scene.env_origins
    return torch.cat([env.scene[name].data.root_pos_w.torch - origins for name in piece_names], dim=-1)


def piece_orientations(env: ManagerBasedRLEnv, piece_names: list[str]) -> torch.Tensor:
    """Orientations of every chess piece as (x, y, z, w) quaternions. Shape is (N, 4 * pieces)."""
    return torch.cat([env.scene[name].data.root_quat_w.torch for name in piece_names], dim=-1)


def commanded_piece_pose(
    env: ManagerBasedRLEnv,
    command_name: str = "chess_move",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Pose of the piece the command selected, plus the vector from the TCP to it.

    Layout is ``[pos (3), quat (4), tcp_to_piece (3)]``; positions are relative to
    the environment origin so the term is translation-invariant across envs.
    """
    command = env.command_manager.get_term(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    piece_pos_w = command.piece_pos_w
    index = command.piece_index
    quats = torch.stack([env.scene[name].data.root_quat_w.torch for name in command.cfg.piece_names], dim=1)
    piece_quat = quats[torch.arange(env.num_envs, device=quats.device), index]

    return torch.cat(
        (
            piece_pos_w - env.scene.env_origins,
            piece_quat,
            piece_pos_w - ee_frame.data.target_pos_w.torch[:, 0, :],
        ),
        dim=-1,
    )


def piece_grasped(
    env: ManagerBasedRLEnv,
    open_command: dict[str, float],
    close_command: dict[str, float],
    command_name: str = "chess_move",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    max_tcp_distance: float = 0.18,
    min_fraction: float = 0.04,
    max_fraction: float = 0.94,
) -> torch.Tensor:
    """Whether the commanded piece is held: fingers part-closed, with the TCP on it.

    Two details matter, and getting either wrong reports *not holding* on a perfectly
    good grasp:

    * The opening is measured as a **fraction of the stroke averaged over the
      fingers**, never per finger. Fingers are independently actuated, so a grasp a
      few millimetres off-centre leaves one nearly shut and the other wide; and the
      raw joint values are not comparable between arms anyway.
    * The TCP-to-piece tolerance has to cover the whole piece, not just its axis. A
      piece's origin sits on the board while it is gripped 45-115 mm up, and a knight
      is gripped off-axis entirely -- 0.18 m clears the tallest grip with margin while
      still being far tighter than any other object in the scene.

    Fingers at either end of their travel mean the gripper shut on nothing or never
    closed, which the fraction bounds reject.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_term(command_name)

    fraction = gripper_open_fraction(env, open_command, close_command, robot_cfg).squeeze(-1)
    pinched = (fraction > min_fraction) & (fraction < max_fraction)

    distance = torch.norm(ee_frame.data.target_pos_w.torch[:, 0, :] - command.piece_pos_w, dim=-1)
    return (pinched & (distance < max_tcp_distance)).unsqueeze(-1)


def piece_lifted(
    env: ManagerBasedRLEnv,
    command_name: str = "chess_move",
    board_height: float = 0.0,
    min_height: float = 0.03,
) -> torch.Tensor:
    """Whether the commanded piece has been raised clear of the board."""
    command = env.command_manager.get_term(command_name)
    height = command.piece_pos_w[:, 2] - env.scene.env_origins[:, 2] - board_height
    return (height > min_height).unsqueeze(-1)


def piece_placed(
    env: ManagerBasedRLEnv,
    command_name: str = "chess_move",
    xy_threshold: float = 0.02,
    height_threshold: float = 0.01,
    max_speed: float = 0.05,
    max_tilt_deg: float = 25.0,
) -> torch.Tensor:
    """Whether the commanded piece is standing still, upright, on its target square."""
    command = env.command_manager.get_term(command_name)
    index = command.piece_index
    arange = torch.arange(env.num_envs, device=index.device)

    piece_pos_w = command.piece_pos_w
    offset = piece_pos_w - command.target_pos_w
    on_square = torch.norm(offset[:, :2], dim=-1) < xy_threshold
    on_surface = offset[:, 2].abs() < height_threshold

    velocities = torch.stack(
        [env.scene[name].data.root_lin_vel_w.torch for name in command.cfg.piece_names], dim=1
    )[arange, index]
    settled = torch.norm(velocities, dim=-1) < max_speed

    quats = torch.stack([env.scene[name].data.root_quat_w.torch for name in command.cfg.piece_names], dim=1)
    piece_up = math_utils.quat_apply(quats[arange, index], torch.tensor([0.0, 0.0, 1.0], device=quats.device).expand(env.num_envs, 3))
    upright = piece_up[:, 2] > torch.cos(torch.deg2rad(torch.tensor(max_tilt_deg, device=quats.device)))

    return (on_square & on_surface & settled & upright).unsqueeze(-1)
