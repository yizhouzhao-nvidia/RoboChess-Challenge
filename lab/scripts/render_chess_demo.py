"""Replay recorded chess demonstrations and render them to images / video.

Loads episodes from the HDF5 written by ``lab/scripts/generate_chess_pick_demos.py``,
restores each episode's initial state and re-executes its recorded actions through
the simulator. Because it replays *actions* rather than forcing states, a clean
render is also evidence that the dataset is self-contained and reproducible.

Outputs, per demo, into ``--output_dir``:

* ``<demo>_filmstrip.png`` -- key frames of the pick-and-place across one image,
* ``<demo>_frames/*.png``  -- the individual frames (with ``--save_frames``),
* ``<demo>.mp4``           -- the full replay (with ``--video``).

.. code-block:: bash

    python lab/scripts/render_chess_demo.py \
        --dataset_file lab/datasets/franka_chess_pick.hdf5 --demos 0 1 2 --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay and render Franka chess demonstrations.")
parser.add_argument("--task", type=str, default="RoboChess-Franka-Chess-IK-Abs-v0")
parser.add_argument(
    "--robot", type=str, default=None, help="Arm the dataset was recorded with: franka, piper, rebot or yam."
)
parser.add_argument("--chess_scenario", choices=("pieces", "1d", "3x3", "4x4", "8x8"), default="4x4")
parser.add_argument(
    "--dataset_file", type=str, default="./lab/datasets/franka_chess_pick.hdf5", help="Recorded dataset to replay."
)
parser.add_argument("--demos", type=int, nargs="+", default=[0], help="Episode indices to replay.")
parser.add_argument("--output_dir", type=str, default="./lab/datasets/renders")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--filmstrip_frames", type=int, default=6, help="Key frames per filmstrip.")
parser.add_argument("--video", action="store_true", help="Also write an mp4 of the whole replay.")
parser.add_argument("--video_stride", type=int, default=1, help="Keep every n-th step in the video.")
parser.add_argument("--save_frames", action="store_true", help="Keep the individual key frames as PNGs.")
parser.add_argument(
    "--view",
    choices=("showcase", "three_quarter", "front", "top"),
    default="three_quarter",
    help="Camera placement relative to the board.",
)
parser.add_argument(
    "--zoom",
    type=float,
    default=1.0,
    help="Scale the camera distance; >1 pulls back (the 8x8 board needs ~1.4).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

import isaaclab.sim as sim_utils
import robochess.tasks  # noqa: F401
from isaaclab.sensors import CameraCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

# Eye / look-at pairs in the environment frame. The board sits in front of the arm
# on the table top and the capture tray is off to -y, so the default view keeps both
# the board and the tray in shot.
VIEWS = {
    # Frames the whole arm, not just the board: the other views tilt down from
    # roughly wrist height and cut the robot off above the elbow, which is fine for
    # watching a grasp but useless for a figure that has to show which robot it is.
    "showcase": ((1.32, -1.18, 1.78), (-0.02, 0.0, 1.02)),
    "three_quarter": ((0.78, -0.66, 1.24), (0.16, -0.12, 0.83)),
    "front": ((0.95, 0.0, 1.06), (0.12, 0.0, 0.84)),
    "top": ((0.26, -0.16, 1.55), (0.22, -0.14, 0.77)),
}

GRIPPER_ACTION_INDEX = 7
"""Column of the recorded action holding the binary gripper command."""


def key_step_indices(actions: torch.Tensor, count: int) -> tuple[list[int], list[str]]:
    """Pick the frames that actually show the manipulation, with captions.

    Uniform sampling wastes most of a filmstrip on the arm travelling. The recorded
    gripper command marks the two moments that matter -- when the fingers close and
    when they open -- so the frames are anchored to those instead.
    """
    num_steps = len(actions)
    gripper = actions[:, GRIPPER_ACTION_INDEX]
    closed = (gripper < 0).nonzero().flatten()
    if len(closed) == 0:
        steps = np.linspace(0, num_steps - 1, count).round().astype(int).tolist()
        return steps, [f"step {s + 1}" for s in steps]

    close_at = int(closed[0])
    reopened = (gripper[close_at:] > 0).nonzero().flatten()
    release_at = close_at + int(reopened[0]) if len(reopened) else num_steps - 1

    moments = [
        (0, "1. start"),
        (close_at - 6, "2. reach"),
        (close_at + 12, "3. grasp"),
        ((close_at + release_at) // 2, "4. carry"),
        (release_at - 4, "5. lower"),
        (num_steps - 1, "6. release"),
    ]
    if count != len(moments):
        indices = np.linspace(0, len(moments) - 1, count).round().astype(int)
        moments = [moments[i] for i in indices]

    # Force the picks strictly apart: on a short episode "lower" and "release" can
    # land on the same step, and a duplicate silently drops a filmstrip cell.
    steps: list[int] = []
    for step, _ in moments:
        step = int(np.clip(step, 0, num_steps - 1))
        if steps and step <= steps[-1]:
            step = min(steps[-1] + 1, num_steps - 1)
        steps.append(step)
    return steps, [caption for _, caption in moments]


def make_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/replay_camera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 12.0)
        ),
        offset=CameraCfg.OffsetCfg(convention="ros"),
    )


def label(frame: np.ndarray, text: str) -> Image.Image:
    """Burn a caption into the top-left corner of a frame."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (max(150, 9 * len(text)), 30)], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return image


def replay_drift(env, episode) -> float:
    """Largest gap [m] between where a piece ended in the replay and in the recording.

    Replaying actions re-simulates the contact, so this is the honest measure of how
    reproducible a recorded episode is -- a small number means the dataset carries
    everything needed to recreate the trajectory.
    """
    recorded = episode.data["states"]["rigid_object"]
    live = env.scene.get_state(is_relative=True)["rigid_object"]
    worst = 0.0
    for name, states in recorded.items():
        final = states["root_pose"][-1][:3]
        actual = live[name]["root_pose"][0][:3]
        worst = max(worst, float(torch.norm(actual - final)))
    return worst


def build_filmstrip(frames: list[Image.Image], columns: int) -> Image.Image:
    """Tile key frames into a single contact sheet."""
    rows = (len(frames) + columns - 1) // columns
    width, height = frames[0].size
    sheet = Image.new("RGB", (columns * width, rows * height), (18, 18, 18))
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % columns) * width, (index // columns) * height))
    return sheet


def main():
    dataset_path = Path(args_cli.dataset_file).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"No dataset at {dataset_path}. Run lab/scripts/generate_chess_pick_demos.py first.")
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    handler = HDF5DatasetFileHandler()
    handler.open(str(dataset_path))
    episode_names = list(handler.get_episode_names())
    print(f"[INFO] {dataset_path.name} holds {len(episode_names)} episodes")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # The arm has to match the one the dataset was recorded with, or the replayed
    # actions drive a different kinematic chain and the pieces never get touched.
    if args_cli.robot:
        env_cfg.set_robot(args_cli.robot)
    env_cfg.set_chess_scenario(args_cli.chess_scenario)
    # Replays are driven purely by the recorded actions, so nothing should end an
    # episode early or nudge the pieces on reset.
    env_cfg.terminations.success = None
    env_cfg.terminations.piece_off_board = None
    env_cfg.terminations.board_disturbed = None
    env_cfg.events.reset_pieces.params["position_noise"] = 0.0
    env_cfg.events.reset_pieces.params["yaw_noise"] = 0.0
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.scene.replay_camera = make_camera_cfg()

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    camera = env.scene["replay_camera"]
    eye, look_at = (np.array(v) for v in VIEWS[args_cli.view])
    eye = look_at + (eye - look_at) * args_cli.zoom
    origin = env.scene.env_origins[0].cpu().numpy()
    camera.set_world_poses_from_view(
        eyes=torch.tensor([eye + origin], dtype=torch.float32, device=env.device),
        targets=torch.tensor([look_at + origin], dtype=torch.float32, device=env.device),
    )

    def grab(text: str) -> Image.Image:
        env.sim.render()
        camera.update(dt=0.0)
        return label(camera.data.output["rgb"][0, ..., :3].cpu().numpy(), text)

    with torch.inference_mode():
        for demo_index in args_cli.demos:
            if demo_index >= len(episode_names):
                print(f"[WARN] demo {demo_index} does not exist, skipping")
                continue
            name = episode_names[demo_index]
            episode = handler.load_episode(name, env.device)
            actions = episode.data["actions"]
            num_steps = len(actions)
            env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)

            key_steps, key_captions = key_step_indices(actions, args_cli.filmstrip_frames)
            captions = dict(zip(key_steps, key_captions))
            key_frames: list[Image.Image] = []
            video_frames: list[Image.Image] = []

            for step in range(num_steps):
                env.step(actions[step].unsqueeze(0))
                wanted_key = step in captions
                wanted_video = args_cli.video and step % args_cli.video_stride == 0
                if not (wanted_key or wanted_video):
                    continue
                caption = captions.get(step, name)
                frame = grab(f"{caption}   t={step * env.step_dt:4.1f}s")
                if wanted_key:
                    key_frames.append(frame)
                if wanted_video:
                    video_frames.append(frame)

            strip = build_filmstrip(key_frames, columns=min(3, len(key_frames)))
            strip_path = output_dir / f"{name}_filmstrip.png"
            strip.save(strip_path)

            drift = replay_drift(env, episode)
            print(f"[INFO] {name}: {num_steps} steps, replay drift {drift * 1000:.1f} mm -> {strip_path}")

            if args_cli.save_frames:
                frame_dir = output_dir / f"{name}_frames"
                frame_dir.mkdir(exist_ok=True)
                for index, frame in enumerate(key_frames):
                    frame.save(frame_dir / f"{index:02d}.png")

            if args_cli.video and video_frames:
                import imageio.v2 as imageio

                video_path = output_dir / f"{name}.mp4"
                fps = max(1, round(1.0 / (env.step_dt * args_cli.video_stride)))
                with imageio.get_writer(video_path, fps=fps) as writer:
                    for frame in video_frames:
                        writer.append_data(np.asarray(frame))
                print(f"[INFO] {name}: video -> {video_path}")

    handler.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
