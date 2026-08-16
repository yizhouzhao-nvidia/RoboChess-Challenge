"""Summarise a recorded chess demonstration dataset.

The authoritative answer to "what is actually in this dataset" -- which pieces were
moved, how accurately, how long the episodes are. Worth having as a separate tool
because a counter kept inside the generation loop cannot be trusted: the environment
auto-resets inside ``step()``, so anything read after the fact describes the *next*
episode. This reads the file.

Needs no Isaac Sim, only h5py:

.. code-block:: bash

    python lab/scripts/dataset_summary.py lab/datasets/chess_pick_rebot.hdf5 --scenario pieces
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from robochess.tasks.manager_based.chess.board import PIECE_KINDS, make_layout  # noqa: E402

CHESS_MOVE_PIECE_INDEX = 3
"""Column of the ``chess_move`` observation holding the commanded piece index."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--scenario", default="pieces", help="Board layout the dataset was recorded on.")
    parser.add_argument("--board_scale", type=float, default=1.4)
    return parser.parse_args()


def summarise(path: Path, kinds: list[str]) -> None:
    with h5py.File(path, "r") as handle:
        episodes = handle["data"]
        counts: dict[str, int] = {}
        lengths, errors = [], []
        for name in episodes:
            episode = episodes[name]
            if not bool(episode.attrs["success"]):
                print(f"  [warn] {name} is flagged unsuccessful but present in the dataset")
            index = int(round(float(episode["obs/chess_move"][0][CHESS_MOVE_PIECE_INDEX])))
            counts[kinds[index]] = counts.get(kinds[index], 0) + 1
            lengths.append(int(episode.attrs["num_samples"]))

            # Final placement error: commanded target vs where the piece ended up,
            # both brought into the robot base frame.
            base = episode["initial_state/articulation/robot/root_pose"][0][:3]
            piece = episode["obs/target_piece"][-1][:3] - base
            errors.append(float(np.linalg.norm(piece[:2] - episode["obs/chess_move"][-1][:2])))

        errors = np.asarray(errors) * 1000.0
        missing = sorted(set(kinds) - set(counts))
        print(f"\n{path.name}: {len(episodes)} demos, {sum(lengths)} transitions, {path.stat().st_size / 1e6:.1f} MB")
        print(f"    episode length : {min(lengths)}-{max(lengths)} steps (mean {np.mean(lengths):.0f})")
        print(f"    placement error: mean {errors.mean():.1f} / p95 {np.percentile(errors, 95):.1f} / max {errors.max():.1f} mm")
        print(f"    pieces moved   : {dict(sorted(counts.items()))}")
        print(f"    kind coverage  : {len(counts)}/{len(set(kinds))}" + (f"  MISSING {missing}" if missing else "  (complete)"))


def main() -> None:
    args = parse_args()
    kinds = [piece.kind for piece in make_layout(args.scenario, args.board_scale).pieces]
    unknown = set(kinds) - set(PIECE_KINDS)
    if unknown:
        raise SystemExit(f"Layout '{args.scenario}' produced unknown piece kinds: {sorted(unknown)}")
    for path in args.datasets:
        if not path.exists():
            print(f"\n{path}: missing")
            continue
        summarise(path, kinds)


if __name__ == "__main__":
    main()
