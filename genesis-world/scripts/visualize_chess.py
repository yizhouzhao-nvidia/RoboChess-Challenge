#!/usr/bin/env python3
"""Visualise a RoboChess scene on genesis-world: one arm, one chess scenario.

The Genesis counterpart of

    python lab/scripts/zero_agent.py --task RoboChess-Visual-v0 --robot franka --chess_scenario 4x4

and of ``newton/scripts/visualize_chess.py``, over the four parallel-jaw arms of the Isaac
Lab picking task (franka, piper, rebot, yam) and all five scenarios.  Nothing is commanded:
the arm holds its home posture on position drives while the pieces settle onto the board,
which is exactly the check this script exists for -- that the assets load, that the arm is
standing on the table and not through the board, and that the pieces are where the layout
says they are.

    python genesis-world/scripts/visualize_chess.py --robot franka --scenario 4x4 --viewer null
    python genesis-world/scripts/visualize_chess.py --robot yam --scenario 8x8 --viewer gl \
        --save-video out/yam_8x8.mp4
    python genesis-world/scripts/visualize_chess.py --world-count 2 --scenario 1d,8x8 --viewer null

``--robot``, ``--scenario`` and ``--board-scale`` take comma-separated lists to give the
worlds different content.  Identical worlds become real Genesis envs; worlds that differ
cannot be, because a batched env is a *copy* of the scene rather than a variant of it, so
they are laid side by side inside a single env -- see
:class:`~robochess_genesis.scene.ChessScene`.

The summary line that matters is ``dz_mm``: how far every piece moved vertically over the
run.  Settled pieces end 1.2-1.7 mm below where they spawn (the 2 mm spawn clearance minus
the convex hulls' undershoot).  A ``dz`` near -300 mm would mean the pieces free-fell; the
script exits non-zero if a NaN appears.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# openusd's Usd.Stage parser segfaults nondeterministically when threaded, and the piece
# transcoder and the Piper USD both go through pxr. Before any import that can reach it.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "genesis-world") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "genesis-world"))

import numpy as np  # noqa: E402

from robochess_genesis import board_layout as bl  # noqa: E402
from robochess_genesis.assets import DEFAULT_VISUAL_FACES  # noqa: E402
from robochess_genesis.robots import ROBOT_OPTIONS  # noqa: E402
from robochess_genesis.scene import WORLD_SPACING, ChessScene  # noqa: E402
from robochess_genesis.viewer_utils import (  # noqa: E402
    FrameCapture,
    add_viewer_args,
    camera_spec_for,
    describe_renderer,
    resolve_viewer,
    run_frame_loop,
    vec3,
    writable_path,
)


def _choice_list(name: str, options: tuple[str, ...]):
    """argparse type for a comma-separated list drawn from *options*.

    ``choices=`` cannot validate a list-valued argument, so the check lives here and the
    error message still names every option.
    """

    def parse(text: str) -> list[str]:
        values = [item.strip() for item in text.split(",") if item.strip()]
        unknown = [value for value in values if value not in options]
        if unknown or not values:
            raise argparse.ArgumentTypeError(
                f"invalid {name} {unknown or [text]}; choose from {', '.join(options)}"
            )
        return values

    return parse


def _float_list(text: str) -> list[float | None]:
    return [None if item.strip() in ("", "auto") else float(item) for item in text.split(",")]


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a count of 1 or more, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--robot",
        type=_choice_list("robot", ROBOT_OPTIONS),
        default=["franka"],
        help=f"Arm to load; comma-separated for per-world arms. One of: {', '.join(ROBOT_OPTIONS)}.",
    )
    parser.add_argument(
        "--scenario",
        type=_choice_list("scenario", bl.SUPPORTED_SCENARIOS),
        default=["4x4"],
        help=f"Chess layout; comma-separated for per-world layouts. One of: {', '.join(bl.SUPPORTED_SCENARIOS)}.",
    )
    parser.add_argument(
        "--board-scale",
        type=_float_list,
        default=[None],
        help="Square stretch factor; 'auto' (default) uses the per-scenario value.",
    )
    parser.add_argument(
        "--world-count",
        type=_positive_int,
        default=None,
        help="Number of scene replicas (default: as many as the longest --robot/--scenario list).",
    )
    parser.add_argument(
        "--world-spacing",
        type=vec3,
        default=None,
        help="Offset between worlds as 'x,y,z' (default: one table width along y).",
    )
    parser.add_argument(
        "--no-visual",
        dest="visual",
        action="store_false",
        help="Skip the render meshes: same physics, faster to build, nothing to see.",
    )
    parser.add_argument(
        "--visual-faces",
        type=int,
        default=DEFAULT_VISUAL_FACES,
        help="Decimate each piece's render mesh to about this many triangles; 0 keeps all 227k."
        " Genesis builds the mesh per entity rather than instancing it, so an 8x8 board at full"
        " resolution is 7.3M triangles. Colliders are never touched.",
    )
    add_viewer_args(parser)
    args = parser.parse_args(argv)

    viewer = resolve_viewer(parser, args)
    if args.num_frames < 1:
        parser.error(f"--num-frames must be >= 1, got {args.num_frames}")
    if args.fps < 1:
        parser.error(f"--fps must be >= 1, got {args.fps}")
    for dest in ("save_images", "save_video"):
        value = getattr(args, dest)
        if value:
            setattr(args, dest, writable_path(parser, "--" + dest.replace("_", "-"), value))
    if args.save_video and Path(args.save_video).suffix.lower() != ".mp4":
        parser.error("--save-video must end in .mp4; Genesis' only video format is h.264 mp4")
    return args, viewer


def piece_heights(scene: ChessScene) -> np.ndarray:
    """Z of every piece in every world, as one flat array."""
    positions, _ = scene.read_link_poses()
    return np.concatenate(
        [positions[world.env, list(world.piece_slots), 2] for world in scene.worlds]
    )


def main(argv: list[str] | None = None) -> int:
    args, viewer = parse_args(argv)

    started = time.perf_counter()
    scene = ChessScene(
        robot=args.robot,
        scenario=args.scenario,
        board_scale=args.board_scale,
        world_count=args.world_count,
        visual=args.visual,
        visual_faces=args.visual_faces,
        world_spacing=args.world_spacing or WORLD_SPACING,
        fps=args.fps,
        camera=camera_spec_for(args, viewer),
        show_viewer=(viewer == "window"),
        backend=args.device,
    )
    built = time.perf_counter()
    scene.finalize()
    finalized = time.perf_counter()

    print(scene.describe(), flush=True)
    print(describe_renderer(scene), flush=True)
    print(
        f"  build={1000 * (built - started):.0f}ms finalize={1000 * (finalized - built):.0f}ms "
        f"substeps/frame={scene.sim_substeps} dt={scene.sim_dt:.5f}",
        flush=True,
    )

    capture = None
    if args.save_images or args.save_video:
        capture = FrameCapture(scene, args.save_images, args.save_video, fps=args.fps)

    start_z = piece_heights(scene)
    loop_started = time.perf_counter()
    frames = run_frame_loop(scene, args.num_frames, capture)
    elapsed = time.perf_counter() - loop_started

    end_z = piece_heights(scene)
    velocities = scene.read_link_velocities()
    # Settled pieces sit ~1.7 mm below where they spawn (the spawn clearance minus the
    # hulls' undershoot). A dz near -300 mm would mean they free-fell.
    dz = 1000.0 * (end_z - start_z)
    nan = bool(np.isnan(end_z).any() or np.isnan(velocities).any())
    print(
        f"  {frames} frames in {elapsed:.2f}s ({1000 * elapsed / max(frames, 1):.1f} ms/frame, "
        f"render included): dz_mm {dz.min():+.2f}..{dz.max():+.2f} "
        f"|v|max={np.abs(velocities).max():.4f} nan={nan}",
        flush=True,
    )
    if capture is not None:
        capture.close()
        capture.report()
    return 1 if nan else 0


if __name__ == "__main__":
    raise SystemExit(main())
