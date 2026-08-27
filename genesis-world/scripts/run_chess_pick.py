#!/usr/bin/env python3
"""Drive GraspGen pick-and-place of chess pieces on genesis-world.

The Genesis counterpart of

    python lab/scripts/generate_chess_pick_demos.py --robot franka --chess_scenario 4x4

and of ``newton/scripts/run_chess_pick.py``, minus the HDF5 recorder: every world is handed
one move per episode -- pick piece *i* up and put it on destination *j* -- and
:class:`~robochess_genesis.pick.ChessPickTask` executes it with the GraspGen grasp that the
board actually leaves room for.  Outcomes are reported per episode and summarised as a
success rate, using the same definition of success as the Isaac Lab task: the piece within
20 mm of its destination square, within 10 mm of the surface, upright to 25 degrees, at rest
and out of the fingers.

    python genesis-world/scripts/run_chess_pick.py --robot franka --scenario 4x4 \
        --world-count 8 --num-episodes 16 --viewer null
    python genesis-world/scripts/run_chess_pick.py --robot franka --scenario 1d \
        --world-count 1 --num-episodes 1 --viewer gl --save-video out/pick.mp4

Episodes, not ``--num-frames``, own the loop here: ``--num-frames`` is an optional cap for
keeping a recording short, and it cuts the run off mid-episode, so a run that used it is a
recording and not a measurement.  All worlds reset together, so a batch of N worlds runs N
episodes at a time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

# openusd's Usd.Stage parser segfaults nondeterministically when threaded, and the piece
# transcoder and the Piper USD both go through pxr. Before any import that can reach it.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "genesis-world") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "genesis-world"))

from robochess_genesis import board_layout as bl  # noqa: E402
from robochess_genesis.assets import DEFAULT_VISUAL_FACES  # noqa: E402
from robochess_genesis.pick import (  # noqa: E402
    PICK_ARM_ARMATURE,
    PICK_ARM_KP,
    PICK_GRIPPER_ARMATURE,
    PICK_GRIPPER_KP,
    ChessPickTask,
    pick_gain_kwargs,
)
from robochess_genesis.robots import ROBOT_OPTIONS  # noqa: E402
from robochess_genesis.scene import ChessScene  # noqa: E402
from robochess_genesis.viewer_utils import (  # noqa: E402
    FrameCapture,
    add_viewer_args,
    camera_spec_for,
    describe_renderer,
    resolve_viewer,
    writable_path,
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a count of 1 or more, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--robot", choices=ROBOT_OPTIONS, default="franka", help="Arm to drive.")
    parser.add_argument(
        "--scenario", choices=bl.SUPPORTED_SCENARIOS, default="4x4", help="Board setup to pick on."
    )
    parser.add_argument(
        "--board-scale", type=float, default=None, help="Square stretch factor (default: per-scenario)."
    )
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
        help="GraspGen candidates per piece. 12 for parity with the Isaac Lab script; the JSON holds"
        " 64 and using all of them mostly helps the knight, the one kind that is not a solid of"
        " revolution and so gets no yaw spin to thread the fingers between neighbours.",
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
        "--arm-ke", type=float, default=PICK_ARM_KP, help="Arm drive stiffness (damping is a twentieth)."
    )
    parser.add_argument(
        "--gripper-ke",
        type=float,
        default=None,
        help="Gripper drive stiffness. Defaults per arm (GenesisRobotSpec.pick_gripper_kp), falling"
        f" back to {PICK_GRIPPER_KP:.0f}; the four grippers are different mechanisms and do not want"
        " the same joint stiffness.",
    )
    parser.add_argument(
        "--arm-armature", type=float, default=PICK_ARM_ARMATURE, help="Rotor inertia on the arm DOFs."
    )
    parser.add_argument(
        "--gripper-armature",
        type=float,
        default=PICK_GRIPPER_ARMATURE,
        help="Rotor inertia on the gripper DOFs.",
    )
    parser.add_argument("--debug", action="store_true", help="Trace world 0 through every control tick.")
    parser.add_argument("--no-visual", dest="visual", action="store_false", help="Drop the render meshes.")
    parser.add_argument(
        "--visual-faces",
        type=int,
        default=DEFAULT_VISUAL_FACES,
        help="Decimate each piece's render mesh to about this many triangles; 0 keeps all 227k.",
    )
    add_viewer_args(parser)
    # The episode count owns the loop; a frame budget is opt-in.
    parser.set_defaults(num_frames=0)
    for action in parser._actions:
        if action.dest == "num_frames":
            action.help = (
                "Frame budget for keeping a recording short; 0 (the default) lets the episodes own"
                " the loop. A budget cuts the run off mid-episode, so never quote the success rate of"
                " a run that used it -- record with it, measure without it."
            )
    args = parser.parse_args(argv)

    viewer = resolve_viewer(parser, args)
    if args.fps < 1:
        parser.error(f"--fps must be >= 1, got {args.fps}")
    if args.num_frames < 0:
        parser.error(f"--num-frames must be >= 0, got {args.num_frames}")
    if args.grasp_file:
        args.grasp_file = str(Path(args.grasp_file).expanduser().resolve())
        if not os.path.exists(args.grasp_file):
            parser.error(
                f"--grasp-file {args.grasp_file} does not exist. The shipped grasps are committed"
                f" through git-lfs (git lfs pull); regenerating them needs GraspGen's own interpreter"
                f" (see lab/Readme.md):\n"
                f"    <graspgen-venv>/bin/python {REPO_ROOT}/lab/scripts/graspgen_chess_grasps.py"
            )
    for dest in ("save_images", "save_video"):
        value = getattr(args, dest)
        if value:
            setattr(args, dest, writable_path(parser, "--" + dest.replace("_", "-"), value))
    if args.save_video and Path(args.save_video).suffix.lower() != ".mp4":
        parser.error("--save-video must end in .mp4; Genesis' only video format is h.264 mp4")
    return args, viewer


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
        fps=args.fps,
        camera=camera_spec_for(args, viewer),
        show_viewer=(viewer == "window"),
        backend=args.device,
    )
    scene.finalize(
        **pick_gain_kwargs(
            args.arm_ke, args.gripper_ke, args.arm_armature, args.gripper_armature, spec=args.robot
        )
    )
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
    print(describe_renderer(scene), flush=True)
    print(f"  setup {time.perf_counter() - started:.1f}s", flush=True)

    capture = None
    if args.save_images or args.save_video:
        capture = FrameCapture(scene, args.save_images, args.save_video, fps=args.fps)

    frame_budget = args.num_frames if args.num_frames > 0 else None
    frames = 0

    def on_frame() -> None:
        nonlocal frames
        if capture is not None:
            capture.grab()
        frames += 1

    # Installed unconditionally. Gating it on "is anything being captured" is the obvious
    # thing and it silently breaks --num-frames, which counts *frames* and so needs the hook
    # even when there is nothing to put in them. The render cost is not attached to the hook:
    # ChessScene decides that from its own camera and viewer.
    frame_hook = on_frame

    attempts = 0
    successes = 0
    failures: Counter[str] = Counter()
    kind_attempts: Counter[str] = Counter()
    kind_successes: Counter[str] = Counter()
    loop_started = time.perf_counter()
    control_steps = 0

    try:
        while attempts < args.num_episodes:
            task.reset_episode()
            while not task.episode_finished:
                task.step(frame_hook)
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
    if task.ik_failures:
        print(
            f"[INFO] IK finished >1mm from the command on at least one world in"
            f" {task.ik_failures} of {control_steps} control ticks"
            " (normal when a leg's target is briefly outside the workspace)",
            flush=True,
        )
    if capture is not None:
        capture.report()
    if not attempts:
        # --num-episodes is validated as >= 1, so this is a run that was stopped early.
        reason = f"the --num-frames {frame_budget} budget" if frame_budget else "the run"
        print(f"error: no episode finished before {reason} ended; nothing was scored", file=sys.stderr)
    return 0 if attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
