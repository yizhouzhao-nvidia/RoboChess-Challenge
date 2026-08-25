#!/usr/bin/env python3
"""Drive GraspGen pick-and-place of chess pieces on newton-physics.

The Newton counterpart of

    python lab/scripts/generate_chess_pick_demos.py --robot franka --chess_scenario 4x4

minus the HDF5 recorder: every world is handed one move per episode -- pick piece
*i* up and put it on destination *j* -- and
:class:`~robochess_newton.pick.ChessPickTask` executes it with the GraspGen grasp
that the board actually leaves room for.  Outcomes are reported per episode and
summarised as a success rate, using the same definition of success as the Isaac Lab
task: the piece within 20 mm of its destination square, within 10 mm of the surface,
upright to 25 degrees, at rest and out of the fingers.

    python newton/scripts/run_chess_pick.py --robot franka --scenario 4x4 \
        --world-count 8 --num-episodes 16 --viewer null
    python newton/scripts/run_chess_pick.py --robot franka --scenario 1d \
        --num-episodes 1 --viewer gl --headless --save-video out/pick.mp4

Episodes, not ``--num-frames``, own the loop here: ``--num-frames`` is an optional
cap for keeping a recording short, and it cuts the run off mid-episode, so a run
that used it is a recording and not a measurement.  All worlds reset together, so
a batch of N worlds runs N episodes at a time.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# openusd's UsdPhysics parser segfaults nondeterministically when threaded, and every
# asset in this scene goes through it. Before any import that can reach pxr.
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

from robochess_newton import board_layout as bl  # noqa: E402
from robochess_newton.pick import (  # noqa: E402
    PICK_ARM_ARMATURE,
    PICK_ARM_KE,
    PICK_GRIPPER_ARMATURE,
    PICK_GRIPPER_KE,
    ChessPickTask,
    apply_pick_gains,
)
from robochess_newton.robots import ROBOT_OPTIONS  # noqa: E402
from robochess_newton.scene import ChessScene  # noqa: E402
from robochess_newton.viewer_utils import add_viewer_args, has_display, make_viewer  # noqa: E402

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--robot", choices=ROBOT_OPTIONS, default="franka", help="Arm to drive.")
    parser.add_argument(
        "--scenario", choices=bl.SUPPORTED_SCENARIOS, default="4x4", help="Board setup to pick on."
    )
    parser.add_argument("--board-scale", type=float, default=None, help="Square stretch factor (default: per-scenario).")
    parser.add_argument("--world-count", type=_positive_int, default=4, help="Worlds stepped in parallel.")
    parser.add_argument(
        "--num-episodes", type=_positive_int, default=16, help="Total move attempts across all worlds."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grasp-file", type=str, default=None, help="Override the GraspGen grasp JSON.")
    parser.add_argument(
        "--num-grasp-candidates",
        type=_positive_int,
        default=12,
        help="GraspGen candidates per piece. 12 for parity with the Isaac Lab script; the JSON"
        " holds 64 and using all of them mostly helps the knight, the one kind that is not a"
        " solid of revolution and so gets no yaw spin to thread the fingers between neighbours"
        " (measured, franka/1d, 24 episodes: 21/24 at 12, 23/24 at 64, for 38 ms of planning per"
        " world per episode instead of 8).",
    )
    parser.add_argument(
        "--num-yaw-candidates", type=_positive_int, default=16, help="Yaw samples for pieces of revolution."
    )
    parser.add_argument(
        "--balance-kinds",
        action="store_true",
        help="Steer each episode toward the least-attempted piece kind, so a short run still covers"
        " pawn, rook, knight, bishop, queen and king.",
    )
    parser.add_argument(
        "--arm-ke", type=float, default=PICK_ARM_KE, help="Arm drive stiffness (damping is a twentieth)."
    )
    parser.add_argument(
        "--gripper-ke", type=float, default=PICK_GRIPPER_KE, help="Gripper drive stiffness."
    )
    parser.add_argument(
        "--arm-armature", type=float, default=PICK_ARM_ARMATURE, help="Rotor inertia on the arm DOFs."
    )
    parser.add_argument(
        "--gripper-armature",
        type=float,
        default=PICK_GRIPPER_ARMATURE,
        help="Rotor inertia on the gripper DOFs. Retained as a default; no case measured so far is sensitive to it (unlike --arm-armature, where 0 does break the arm).",
    )
    parser.add_argument("--debug", action="store_true", help="Trace world 0 through every control tick.")
    parser.add_argument("--no-visual", dest="visual", action="store_false", help="Drop the render meshes.")
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
        "Frame budget for keeping a recording short; 0 (the default) lets the episodes own the loop."
        " A budget cuts the run off mid-episode, so never quote the success rate of a run that used"
        " it -- record with it, measure without it.",
    )
    _set_help(
        parser,
        "output_path",
        "Artifact written by the usd/file/rerun/viser viewers; the extension has to match the"
        " backend (.usd/.usda/.usdc, .bin/.json, .rrd, .viser). Default:"
        f" {DEFAULT_OUTPUT_DIR}/pick_<robot>_<scenario>.usd, overridable with $ROBOCHESS_OUT_DIR.",
    )
    # The episode count owns the loop; a frame budget is opt-in.
    parser.set_defaults(num_frames=0)
    args = parser.parse_args(argv)
    _validate(parser, args)
    return args


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
    return f"pick_{args.robot}_{args.scenario}"


def _resolved_viewer(args: argparse.Namespace) -> str:
    """The backend :func:`make_viewer` will build; mirrors its auto policy.

    Needed at parse time, where the viewer does not exist yet, to check
    ``--output-path`` against the backend that will honour it.
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

    if args.grasp_file:
        args.grasp_file = str(Path(args.grasp_file).expanduser().resolve())
        if not os.path.exists(args.grasp_file):
            # Same remediation as GraspLibrary's, but absolute: it is as likely to be
            # read from a log as from a shell sitting in the repo root.
            parser.error(
                f"--grasp-file {args.grasp_file} does not exist. Regenerate the grasps with"
                f" GraspGen's own interpreter (see lab/Readme.md):\n"
                f"    <graspgen-venv>/bin/python {REPO_ROOT}/lab/scripts/graspgen_chess_grasps.py"
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    viewer, capture = make_viewer(args, default_stem=_stem(args))
    if args.num_frames <= 0 and hasattr(viewer, "num_frames"):
        # ViewerNull stops itself after 1000 frames; here the episodes decide.
        viewer.num_frames = 1 << 30

    started = time.perf_counter()
    scene = ChessScene(
        robot=args.robot,
        scenario=args.scenario,
        board_scale=args.board_scale,
        world_count=args.world_count,
        visual=args.visual,
        fps=args.fps,
    )
    apply_pick_gains(scene, args.arm_ke, args.gripper_ke, args.arm_armature, args.gripper_armature)
    scene.finalize(device=args.device)
    scene.attach_viewer(viewer, eye=args.camera, target=args.camera_target, zoom=args.zoom)
    task = ChessPickTask(
        scene,
        grasp_file=args.grasp_file,
        num_grasp_candidates=args.num_grasp_candidates,
        num_yaw_candidates=args.num_yaw_candidates,
        seed=args.seed,
        balance_kinds=args.balance_kinds,
        debug=args.debug,
    )
    print(scene.describe(), flush=True)
    print(task.describe(), flush=True)
    print(f"  setup {time.perf_counter() - started:.1f}s", flush=True)

    frame_budget = args.num_frames if args.num_frames > 0 else None
    frames = 0

    def on_frame() -> None:
        nonlocal frames
        scene.render()
        if capture is not None:
            capture.grab(frames)
        frames += 1

    attempts = 0
    successes = 0
    failures: Counter[str] = Counter()
    kind_attempts: Counter[str] = Counter()
    kind_successes: Counter[str] = Counter()
    loop_started = time.perf_counter()
    control_steps = 0

    try:
        while attempts < args.num_episodes and viewer.is_running():
            task.reset_episode()
            while not task.episode_finished:
                task.step(on_frame)
                control_steps += 1
                if frame_budget is not None and frames >= frame_budget:
                    break
            for result in task.collect_results():
                if attempts >= args.num_episodes:
                    break
                attempts += 1
                kind_attempts[result.kind] += 1
                if result.success:
                    successes += 1
                    kind_successes[result.kind] += 1
                else:
                    failures[result.outcome] += 1
                print(
                    f"[episode {attempts:3d}] world {result.world} {result.kind:<6s} {result.piece:<18s}"
                    f" -> ({result.target[0]:+.3f},{result.target[1]:+.3f})"
                    f"  {result.outcome:<15s} phase={result.phase:<9s} steps={result.steps:3d}"
                    f" place_err={result.place_error * 1000:6.1f}mm"
                    f" grasp={result.grasp_score:.3f}/pen={result.grasp_penetration * 1000:.1f}mm",
                    flush=True,
                )
            if frame_budget is not None and frames >= frame_budget:
                print(f"[INFO] frame budget {frame_budget} reached", flush=True)
                break
    except KeyboardInterrupt:
        print("[WARN] interrupted", flush=True)
    finally:
        if capture is not None:
            capture.close()
        viewer.close()

    elapsed = time.perf_counter() - loop_started
    rate = successes / attempts if attempts else 0.0
    print(
        f"\n[INFO] {successes} successes / {attempts} attempts ({rate:.0%})"
        f" in {elapsed:.1f}s ({control_steps} control ticks, {frames} frames)",
        flush=True,
    )
    if failures:
        print(
            "[INFO] failures by cause: " + ", ".join(f"{k}={v}" for k, v in sorted(failures.items())),
            flush=True,
        )
    if kind_attempts:
        print(
            "[INFO] by piece kind: "
            + ", ".join(f"{k}={kind_successes[k]}/{kind_attempts[k]}" for k in sorted(kind_attempts)),
            flush=True,
        )
    outputs_ok = _report_outputs(viewer, args)
    if not attempts:
        # --num-episodes is validated as >= 1, so this is a run that was stopped early.
        reason = f"the --num-frames {frame_budget} budget" if frame_budget else "the run"
        print(f"error: no episode finished before {reason} ended; nothing was scored", file=sys.stderr)
    return 0 if attempts and outputs_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
