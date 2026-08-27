#!/usr/bin/env python3
"""Prove that this port's grasp math is the Newton port's, by running both and diffing.

``grasps.py`` is the half of each port that has no engine in it: GraspGen JSON in, numpy
out, no ``newton`` and no ``genesis`` import anywhere on the path.  It is therefore the one
part of the translation that can be checked *exactly* rather than statistically -- and it is
worth checking exactly, because it is also where a silent error is most expensive.  A
quaternion convention flipped in :mod:`robochess_genesis.gsmath`, a transposed rotation in
``graspgen_to_ee``, a sign lost in the probe cloud: none of those crash, none of them show up
in a rendered frame, and all of them come out as a pick success rate that is merely a bit
worse than the reference.

So this script dumps a fingerprint of the planner -- the per-arm retargeting matrix, the
24-point gripper probe cloud, the carry heights, and for every piece kind the chosen
candidate, its score, its penetration and all seven derived keypoints -- against a fixed
synthetic board.  Run it under both interpreters and diff.

    NEWTON=/home/yizhou/Projects/newton/.venv/bin/python
    GENESIS=/home/yizhou/Projects/genesis/.venv/bin/python
    S=genesis-world/scripts/crosscheck_grasps.py

    $NEWTON  $S --port newton  > /tmp/newton.json
    $GENESIS $S --port genesis > /tmp/genesis.json
    diff /tmp/newton.json /tmp/genesis.json && echo IDENTICAL

Two interpreters rather than one because the two packages cannot be imported into the same
process: ``robochess_newton`` needs ``newton``/``warp`` and ``robochess_genesis`` needs
``genesis``, and the two stacks pin incompatible dependencies.  Last run: **identical**, all
four arms, all six kinds.

``--port newton`` also strips the repo root from ``sys.path`` before importing, because this
repository has a top-level ``newton/`` directory that would otherwise shadow the newton
package -- the same guard ``robochess_newton.board_layout`` installs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

ARMS = ("franka", "piper", "rebot", "yam")

BOARD_CENTER = np.array([0.22, 0.0, 0.77])
"""Where the commanded piece stands, in the fixture board below."""

NEIGHBOUR_OFFSETS = ((0.084, 0.0), (-0.084, 0.0), (0.0, 0.084), (0.0, -0.084))
"""One square away in each direction, at the 1.4x-stretched 60 mm pitch."""

NEIGHBOUR_KINDS = ("queen", "pawn", "king", "rook")
"""Deliberately the tall kinds as well as the short ones, so the clearance penalty is live:
a candidate that rakes the fingers through the king has to lose to one that does not."""

PLACE_TARGET = np.array([0.30, -0.10, 0.77])


def load_port(port: str):
    """Import one port's ``grasps`` module and robot table."""
    if port == "newton":
        # The repo's own newton/ directory would shadow the newton package.
        sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != str(REPO_ROOT)]
        sys.path.insert(0, str(REPO_ROOT / "newton"))
        from robochess_newton import grasps
        from robochess_newton.robots import ROBOTS
    elif port == "genesis":
        sys.path.insert(0, str(REPO_ROOT / "genesis-world"))
        from robochess_genesis import grasps
        from robochess_genesis.robots import ROBOTS
    else:
        raise SystemExit(f"unknown port {port!r}; choose 'newton' or 'genesis'")
    return grasps, ROBOTS


def fingerprint(grasps, spec, digits: int) -> dict:
    """Everything the planner derives for one arm, rounded to *digits*."""
    library = grasps.GraspLibrary(grasps.default_grasp_file(), spec, max_candidates=12)
    geometry = grasps.load_piece_geometry()
    planner = grasps.GraspPlanner(spec, library, geometry, num_yaws=16)

    piece_pose = grasps.pose_to_matrix(BOARD_CENTER, np.array([0.0, 0.0, 0.0, 1.0]))
    others = np.array([[BOARD_CENTER[0] + dx, BOARD_CENTER[1] + dy, BOARD_CENTER[2]] for dx, dy in NEIGHBOUR_OFFSETS])

    plans = {}
    for kind in sorted(geometry):
        plan = planner.plan(kind, piece_pose, others, list(NEIGHBOUR_KINDS), 0.12)
        _, place = grasps.place_goals(PLACE_TARGET, piece_pose, plan.hand_pose_w, 0.12)
        plans[kind] = {
            "candidate": plan.candidate,
            "score": round(plan.score, digits),
            "penetration": round(plan.penetration, digits),
            "goals": {n: [np.round(v, digits).tolist() for v in g] for n, g in sorted(plan.goals.items())},
            "place": {n: [np.round(v, digits).tolist() for v in g] for n, g in sorted(place.items())},
        }
    return {
        "convention_fix": np.round(library.convention_fix, digits).tolist(),
        "probe_points": np.round(planner.probe_points, digits).tolist(),
        "carry": [round(grasps.carry_height(geometry[k]["height"], spec.reach), digits) for k in sorted(geometry)],
        "plans": plans,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, choices=("newton", "genesis"), help="Which package to fingerprint.")
    parser.add_argument(
        "--digits",
        type=int,
        default=9,
        help="Rounding before comparison. 9 is well inside float64 but outside the last bit, so a"
        " genuine algebraic difference shows and float reassociation does not.",
    )
    args = parser.parse_args(argv)

    grasps, robots = load_port(args.port)
    print(json.dumps({arm: fingerprint(grasps, robots[arm], args.digits) for arm in ARMS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
