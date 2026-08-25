#!/usr/bin/env python3
"""Visualise a RoboChess scene on newton-physics: one arm, one chess scenario.

The Newton counterpart of

    python lab/scripts/zero_agent.py --task RoboChess-Visual-v0 --robot franka --chess_scenario 4x4

over the four parallel-jaw arms of the Isaac Lab picking task (franka, piper,
rebot, yam) and all five scenarios.  Nothing is commanded: the arm holds
its home posture on position drives while the pieces settle onto the board, which
is exactly the check this script exists for -- that the assets load, that the arm
is standing on the table and not through the board, and that the pieces are where
the layout says they are.

    python newton/scripts/visualize_chess.py --robot franka --scenario 4x4
    python newton/scripts/visualize_chess.py --robot yam --scenario 8x8 --viewer gl --headless \
        --save-video out/yam_8x8.mp4
    python newton/scripts/visualize_chess.py --world-count 2 --scenario 1d,8x8 --viewer null

``--robot``, ``--scenario`` and ``--board-scale`` take comma-separated lists to
give the worlds different content.  Identical worlds are stamped out with
``ModelBuilder.replicate``; worlds that differ cannot be, because ``SolverMuJoCo``
requires homogeneous worlds, so they are laid side by side inside a single newton
world with ``add_builder`` -- see :class:`~robochess_newton.scene.ChessScene`.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# openusd 25.11's UsdPhysics parser segfaults nondeterministically when threaded, and
# every asset in this scene goes through it. Before any import that can reach pxr.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]

# This repository has a top-level ``newton/`` directory -- the one holding the package
# below -- and the repo root lands on sys.path routinely (running a script from it,
# PYTHONPATH=., pytest). Drop it before anything imports the real newton package, and
# put the package directory on in its place so the script runs from any working
# directory. robochess_newton.board_layout re-checks this at import time.
sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != str(REPO_ROOT)]
if str(REPO_ROOT / "newton") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "newton"))

import numpy as np  # noqa: E402

from robochess_newton import board_layout as bl  # noqa: E402
from robochess_newton.robots import ROBOT_OPTIONS  # noqa: E402
from robochess_newton.scene import WORLD_SPACING, ChessScene  # noqa: E402
from robochess_newton.viewer_utils import (  # noqa: E402
    add_viewer_args,
    has_display,
    make_viewer,
    run_viewer_loop,
)

# Where an artifact goes when --output-path is omitted. A cwd-relative "out/" lands in
# whatever tree the caller happens to be standing in -- including an untracked out/ in
# the repo root -- so the default is absolute and cwd-independent.
DEFAULT_OUTPUT_DIR = Path(os.environ.get("ROBOCHESS_OUT_DIR") or Path(tempfile.gettempdir()) / "robochess-newton")

# What each backend can actually write, first entry being the default. ViewerFile
# rejects anything but .json/.bin *after* recording every frame, printing the refusal
# and exiting 0; the usd/rerun/viser backends are no better behaved.
ARTIFACT_EXTENSIONS = {
    "usd": (".usd", ".usda", ".usdc"),
    "file": (".bin", ".json"),
    "rerun": (".rrd",),
    "viser": (".viser",),
    # gl renders to a window or to --save-images/--save-video, but falls back to
    # ViewerUSD when it cannot open one, and that fallback writes --output-path.
    "gl": (".usd", ".usda", ".usdc"),
}


def _choice_list(name: str, options: tuple[str, ...]):
    """argparse type for a comma-separated list drawn from *options*.

    ``choices=`` cannot validate a list-valued argument, so the check lives here and
    the error message still names every option.
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


def _vec3(text: str) -> tuple[float, float, float]:
    parts = [float(item) for item in text.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'x,y,z', got {text!r}")
    return parts[0], parts[1], parts[2]


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a count of 1 or more, got {value}")
    return value


def _set_help(parser: argparse.ArgumentParser, dest: str, help_text: str) -> None:
    """Reword one of the flags :func:`add_viewer_args` owns; it has no hook for it."""
    for action in parser._actions:
        if action.dest == dest:
            action.help = help_text
            return
    raise AssertionError(f"add_viewer_args no longer defines a {dest!r} flag")


def _stem(args: argparse.Namespace) -> str:
    """Filename stem for the artifact this run would write."""
    return f"chess_{'-'.join(args.robot)}_{'-'.join(args.scenario)}"


def _resolved_viewer(args: argparse.Namespace) -> str:
    """The backend :func:`make_viewer` will build; mirrors its auto policy.

    Needed at parse time, where the viewer does not exist yet, to check
    ``--output-path`` and ``--num-frames`` against the backend that will honour them.
    """
    if args.viewer != "auto":
        return args.viewer
    return "gl" if has_display() or args.save_images or args.save_video else "usd"


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject input that would otherwise fail mid-run, silently or not at all."""
    kind = _resolved_viewer(args)
    extensions = ARTIFACT_EXTENSIONS.get(kind, ())

    if args.fps < 1:
        parser.error(f"--fps must be >= 1, got {args.fps}")
    if args.num_frames < 0:
        parser.error(f"--num-frames must be >= 0, got {args.num_frames}")
    if args.num_frames == 0 and (kind in ("usd", "file") or (kind == "gl" and _headless(args))):
        # 0 hands the stop condition to the viewer and these three have none: measured,
        # --viewer usd --num-frames 0 was still rendering at a 60 s timeout.
        parser.error(
            f"--num-frames 0 waits for the viewer to stop the run, and --viewer {kind} never does;"
            " pass a positive frame count"
        )

    if args.output_path:
        if not extensions:
            parser.error(f"--viewer {kind} writes no file, so --output-path cannot be honoured")
        if Path(args.output_path).suffix not in extensions:
            parser.error(
                f"--viewer {kind} cannot write {args.output_path!r}: --output-path must end in "
                + ", ".join(extensions)
            )
        args.output_path = _writable_path(parser, "--output-path", args.output_path)
    elif kind in ("usd", "file", "gl"):
        # Only the backends make_viewer already defaults a path for: an --output-path
        # switches rerun from serving a web viewer to recording an rrd, and viser from
        # a live preview to a recording, so leaving theirs unset is the behaviour.
        args.output_path = str(DEFAULT_OUTPUT_DIR / f"{_stem(args)}{extensions[0]}")

    for dest in ("save_images", "save_video"):
        value = getattr(args, dest)
        if value:
            setattr(args, dest, _writable_path(parser, "--" + dest.replace("_", "-"), value))

    if args.device:
        import warp as wp

        try:
            wp.get_device(args.device)
        except Exception as exc:  # warp raises a bare ValueError from inside make_viewer
            parser.error(
                f"--device {args.device!r} is not usable ({exc}); this machine has "
                + ", ".join(str(device) for device in wp.get_devices())
            )


def _writable_path(parser: argparse.ArgumentParser, flag: str, path: str) -> str:
    """*path* made absolute, refused now if nothing can be created there.

    Absolute so that every path the run prints -- including the ones viewer_utils
    logs -- is one the reader can open from anywhere; checked because the makedirs
    inside viewer_utils raises PermissionError halfway through an otherwise good run.
    """
    target = Path(path).expanduser().resolve()
    anchor = next(parent for parent in (target, *target.parents) if parent.is_dir())
    if not os.access(anchor, os.W_OK | os.X_OK):
        parser.error(f"{flag} {target}: {anchor} is not writable")
    return str(target)


def _headless(args: argparse.Namespace) -> bool:
    """Whether the GL viewer would run offscreen; mirrors :func:`make_viewer`."""
    return not has_display() if args.headless is None else bool(args.headless)


def _report_outputs(viewer, args: argparse.Namespace) -> bool:
    """Print the absolute path of every artifact on stdout; False if one is missing."""
    name = type(viewer).__name__
    if name in ("ViewerUSD", "ViewerFile", "ViewerViser"):
        expected = args.output_path
    elif name == "ViewerRerun":
        expected = None if args.rerun_address else args.output_path
    else:
        expected = None

    written = []
    if args.save_video:
        # FrameCapture degrades an mp4 to PNGs in <name>_frames when imageio has no
        # video backend, which is what the newton 1.6 venv does.
        written += [args.save_video, os.path.splitext(args.save_video)[0] + "_frames"]
    if args.save_images:
        written.append(args.save_images)
    for path in written:
        if os.path.exists(path):
            print(f"[output] wrote {path}", flush=True)

    if expected is None:
        return True
    if os.path.exists(expected):
        print(f"[output] wrote {expected}", flush=True)
        return True
    print(f"error: {name} wrote no artifact at {expected}", file=sys.stderr)
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        type=_vec3,
        default=None,
        help="Offset between worlds as 'x,y,z' (default: one table width along y).",
    )
    parser.add_argument(
        "--no-visual",
        dest="visual",
        action="store_false",
        help="Skip the render meshes: same physics, ~300 ms less finalize on 8x8, nothing to see.",
    )
    parser.add_argument("--zoom", type=float, default=1.0, help="Camera distance divisor.")
    # A negative x reads as an option, so these need --camera=-1,0,2 rather than a space.
    parser.add_argument("--camera", type=_vec3, default=None, help="Camera eye position '--camera=x,y,z'.")
    parser.add_argument(
        "--camera-target", type=_vec3, default=None, help="Camera look-at point '--camera-target=x,y,z'."
    )
    add_viewer_args(parser)
    _set_help(
        parser,
        "num_frames",
        "Frames to simulate and render; 0 leaves the stop condition to the viewer, which only an"
        " on-screen gl/viser/rerun session supplies (--viewer null stops itself at 1000 frames;"
        " usd, file and headless gl need a positive count).",
    )
    _set_help(
        parser,
        "output_path",
        "Artifact written by the usd/file/rerun/viser viewers; the extension has to match the"
        " backend (.usd/.usda/.usdc, .bin/.json, .rrd, .viser). Default:"
        f" {DEFAULT_OUTPUT_DIR}/chess_<robot>_<scenario>.usd, overridable with $ROBOCHESS_OUT_DIR.",
    )
    args = parser.parse_args(argv)
    _validate(parser, args)
    return args


def piece_heights(scene: ChessScene) -> np.ndarray:
    """Z of every piece body in every world, as one flat array."""
    body_q = scene.state_0.body_q.numpy()
    return np.concatenate([body_q[list(world.piece_bodies), 2] for world in scene.worlds])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    viewer, capture = make_viewer(args, default_stem=_stem(args))

    started = time.perf_counter()
    scene = ChessScene(
        robot=args.robot,
        scenario=args.scenario,
        board_scale=args.board_scale,
        world_count=args.world_count,
        visual=args.visual,
        world_spacing=args.world_spacing or WORLD_SPACING,
        fps=args.fps,
    )
    built = time.perf_counter()
    scene.finalize(device=args.device)
    finalized = time.perf_counter()

    print(scene.describe(), flush=True)
    print(
        f"  build={1000 * (built - started):.0f}ms finalize={1000 * (finalized - built):.0f}ms "
        f"substeps/frame={scene.sim_substeps} dt={scene.sim_dt:.5f}",
        flush=True,
    )

    scene.attach_viewer(viewer, eye=args.camera, target=args.camera_target, zoom=args.zoom)

    start_z = piece_heights(scene)
    loop_started = time.perf_counter()
    frames = run_viewer_loop(viewer, scene, args, capture)
    elapsed = time.perf_counter() - loop_started

    end_z = piece_heights(scene)
    velocities = scene.state_0.body_qd.numpy()
    # Settled pieces sit ~1.7 mm below where they spawn (the spawn clearance minus
    # the hulls' undershoot). A dz near -300 mm means the contact budget was too
    # small and every piece free-fell, which is the failure this line exists to catch.
    dz = 1000.0 * (end_z - start_z)
    nan = bool(np.isnan(end_z).any() or np.isnan(velocities).any())
    print(
        f"  {frames} frames in {elapsed:.2f}s ({1000 * elapsed / max(frames, 1):.1f} ms/frame, "
        f"render included): dz_mm {dz.min():+.2f}..{dz.max():+.2f} "
        f"|qd|max={np.abs(velocities).max():.4f} nan={nan} "
        f"contacts={int(scene.contacts.rigid_contact_count.numpy()[0])}",
        flush=True,
    )
    return 0 if _report_outputs(viewer, args) and not nan else 1


if __name__ == "__main__":
    raise SystemExit(main())
