"""Render hero stills of the two-arm game scene, for the CoRL teaser."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=1000)
parser.add_argument("--out", type=str, default="./lab/docs/images")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pathlib import Path
import gymnasium as gym, numpy as np, torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg
import robochess.tasks  # noqa: F401

VIEWS = {
    "hero":  ((0.0, -1.85, 1.60), (0.0, 0.0, 0.92)),
    "angle": ((-1.15, -1.35, 1.50), (0.0, 0.0, 0.88)),
}

def camera_cfg():
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/hero_camera", update_period=0.0,
        height=args_cli.height, width=args_cli.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 12.0)),
        offset=CameraCfg.OffsetCfg(convention="ros"),
    )

out = Path(args_cli.out); out.mkdir(parents=True, exist_ok=True)
for scenario, pair in (("minichess", ("franka", "franka")),
                       ("1d", ("franka", "franka")),
                       ("3x3", ("franka", "piper"))):
    cfg = parse_env_cfg("RoboChess-Chess-Game-IK-Abs-v0", device=args_cli.device, num_envs=1)
    cfg.set_players(*pair)
    cfg.set_chess_scenario(scenario)
    cfg.scene.hero_camera = camera_cfg()
    env = gym.make("RoboChess-Chess-Game-IK-Abs-v0", cfg=cfg).unwrapped
    env.reset()
    for _ in range(45):
        env.sim.step(); env.scene.update(env.physics_dt)
    cam = env.scene["hero_camera"]
    origin = env.scene.env_origins[0].cpu().numpy()
    for view, (eye, look) in VIEWS.items():
        cam.set_world_poses_from_view(
            eyes=torch.tensor([np.array(eye) + origin], dtype=torch.float32, device=env.device),
            targets=torch.tensor([np.array(look) + origin], dtype=torch.float32, device=env.device))
        for _ in range(6):
            env.sim.render(); cam.update(0.0)
        frame = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        path = out / f"hero_{scenario}_{view}.png"
        Image.fromarray(frame).save(path)
        print("wrote", path)
    env.close()
simulation_app.close()
