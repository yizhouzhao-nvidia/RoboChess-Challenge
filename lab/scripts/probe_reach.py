"""Can this arm reach the board, and how accurately does its IK track?

The second half of adding a robot (see ``probe_robot.py`` for the first). Drives the
task's own differential-IK action to a grid of top-down poses over the board and
reports the steady-state error at each, which answers three questions at once:

* is the home posture sane, or does the arm start folded into the table?
* which squares are actually reachable, i.e. what ``ChessRobotSpec.reach`` should be?
* is the position gain high enough? A chess piece is gripped on a 19-43 mm shaft, so
  anything above ~5 mm of steady-state error will miss.

.. code-block:: bash

    python lab/scripts/probe_reach.py --robot rebot --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check an arm's reach over the chess board.")
parser.add_argument("--robot", type=str, default="franka")
parser.add_argument("--chess_scenario", choices=("pieces", "1d", "3x3", "4x4", "8x8"), default="8x8")
parser.add_argument("--settle_steps", type=int, default=150, help="Control steps to hold each target.")
parser.add_argument("--grasp_height", type=float, default=0.09, help="TCP height above the board [m].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
import robochess.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from robochess.tasks.manager_based.chess.franka_chess_env_cfg import TABLE_TOP_Z

TASK = "RoboChess-Chess-Pick-IK-Abs-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.set_robot(args_cli.robot)
    env_cfg.set_chess_scenario(args_cli.chess_scenario)
    # Nothing should end an episode while we are probing poses.
    env_cfg.terminations.success = None
    env_cfg.terminations.board_disturbed = None
    env_cfg.terminations.piece_off_board = None
    env_cfg.episode_length_s = 1.0e6

    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    spec = env_cfg.chess_robot()
    layout = env_cfg.chess_layout()
    board_center = env_cfg.board_center()
    robot = env.scene["robot"]
    ee_index = robot.find_bodies(spec.ee_body)[0][0]

    def ee_pose_b():
        return math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch,
            robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, ee_index],
            robot.data.body_quat_w.torch[:, ee_index],
        )

    start_pos, _ = ee_pose_b()
    print(f"\n=== {args_cli.robot}: base at {spec.base_pos}, home TCP-ish at {start_pos[0].cpu().numpy().round(3)}")

    # A top-down grasp: the gripper's approach axis pointing straight down. Build the
    # ee orientation that achieves it, then ask for it over a spread of board squares.
    approach = np.asarray(spec.approach_axis, dtype=float)
    closing = np.asarray(spec.closing_axis, dtype=float)
    columns = np.column_stack([closing, np.cross(approach, closing), approach])
    # world axes we want those to land on: approach -> -Z (down), closing -> +Y
    world = np.column_stack([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    rotation = world @ columns.T
    quat = math_utils.quat_from_matrix(torch.tensor(rotation, dtype=torch.float32, device=env.device).unsqueeze(0))

    ranks = [0, layout.num_ranks // 2, layout.num_ranks - 1]
    files = [0, layout.num_files // 2, layout.num_files - 1]
    results = []
    with torch.inference_mode():
        for rank in ranks:
            for file in files:
                bx, by = layout.square_center(file, rank)
                target_env = torch.tensor(
                    [[board_center[0] + bx, board_center[1] + by, TABLE_TOP_Z + args_cli.grasp_height]],
                    device=env.device,
                )
                target_b, _ = math_utils.subtract_frame_transforms(
                    robot.data.root_pos_w.torch - env.scene.env_origins, robot.data.root_quat_w.torch, target_env
                )
                # The IK drives the ee *body*, but the target is where the TCP should
                # land, so back the command off by the tool offset. Commanding the body
                # straight to the TCP target asks the gripper to go through the table.
                tcp_in_base = math_utils.quat_apply(quat, torch.tensor([spec.tcp_offset], device=env.device))
                ee_target_b = target_b - tcp_in_base
                action = torch.cat([ee_target_b, quat, torch.ones(1, 1, device=env.device)], dim=-1)
                for _ in range(args_cli.settle_steps):
                    env.step(action)
                pos, rot = ee_pose_b()
                # Score the TCP, which is what has to land on the square.
                tcp_b = pos + math_utils.quat_apply(rot, torch.tensor([spec.tcp_offset], device=env.device))
                pos_err = float(torch.norm(tcp_b - target_b))
                rot_err = math.degrees(float(math_utils.quat_error_magnitude(rot, quat)))
                distance = float(np.linalg.norm(np.array([board_center[0] + bx, board_center[1] + by]) - np.array(spec.base_pos[:2])))
                results.append((file, rank, distance, pos_err, rot_err))
                delta = (tcp_b - target_b)[0].cpu().numpy()
                print(
                    f"    square (f{file}, r{rank})  {distance:.2f} m from base:"
                    f"  pos_err={pos_err * 1000:7.1f} mm  rot_err={rot_err:6.2f} deg"
                    f"  short-by={np.round(delta * 1000, 1)}"
                )

    good = [r for r in results if r[3] < 0.006 and r[4] < 3.0]
    print(f"\n{len(good)}/{len(results)} probe poses tracked to within 6 mm / 3 deg")
    if good:
        print(f"    reachable out to {max(r[2] for r in good):.2f} m  (spec.reach = {spec.reach})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
