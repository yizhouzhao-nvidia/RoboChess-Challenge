"""Render one RoboChess robot and chess setup to a PNG image."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render a RoboChess scene.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric scene I/O.")
parser.add_argument("--robot", choices=("so101", "piper", "ur10", "flexiv_rizon", "rebot", "yam"), required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from PIL import Image

import isaaclab.sim as sim_utils
import robochess.tasks  # noqa: F401
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(
        "RoboChess-Visual-v0",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.set_robot(args_cli.robot)
    env_cfg.set_chess_scenario("1d")
    env_cfg.scene.render_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/render_camera",
        update_period=0.0,
        height=720,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(convention="ros"),
    )
    env = gym.make("RoboChess-Visual-v0", cfg=env_cfg)
    env.reset()
    camera = env.unwrapped.scene["render_camera"]
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[1.55, 1.25, 1.35]], device=env.unwrapped.device),
        targets=torch.tensor([[0.05, 0.0, 0.78]], device=env.unwrapped.device),
    )
    with torch.inference_mode():
        for _ in range(8):
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

    rgb = env.unwrapped.scene["render_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args_cli.output)
    print(f"[INFO]: Saved screenshot to {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
