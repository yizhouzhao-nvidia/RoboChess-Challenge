"""Ask GraspGen how a Franka Panda should grasp each chess piece.

Runs the ``franka_panda`` GraspGen model (diffusion generator + discriminator) on
the baked chess-piece meshes from :mod:`prepare_chess_assets`, then filters the
raw predictions down to grasps that are actually executable on a piece standing
on a board:

* the gripper must approach from above (a chess piece is never reachable from
  below -- the table is there),
* no part of the gripper may dip below the board surface,
* the piece must fit inside the finger stroke where the fingers actually close,
* the grasp must survive the discriminator score threshold.

Surviving grasps are then ranked with a penalty on approach tilt. GraspGen's own
argmax is almost always a near-horizontal side grasp of the shaft: excellent in
isolation, and the worst possible choice on a populated board.

Chess pieces except the knight are solids of revolution, so every grasp has a
free rotation about the piece axis.  Surviving grasps are therefore rotated into
a canonical azimuth and the free yaw is handed to the motion generator, which
picks whichever yaw suits the arm.

Results land in ``<asset-dir>/grasps/chess_grasps.json`` plus a per-piece PNG.

This script needs the **GraspGen** environment, not the Isaac Lab one:

.. code-block:: bash

    /home/yizhou/Projects/GraspGen/.venv/bin/python lab/scripts/graspgen_chess_grasps.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tra

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_DIR = REPO_ROOT / "assets" / "chess" / "generated" / "s150"
DEFAULT_GRIPPER_CONFIG = Path("/home/yizhou/Projects/GraspGenModels/checkpoints/graspgen_franka_panda.yml")

GRIPPER_NAME = "franka_panda"

CONVENTION_NOTE = (
    "Grasp matrices are 4x4 row-major in the piece frame (z=0 board plane, +Z up the piece axis)."
    " They follow the GraspGen gripper convention: +Z is the approach direction and +X is the finger"
    " closing direction, with the origin at the gripper base link (Franka panda_hand) and the TCP at"
    " +Z * depth. The Isaac Lab Franka closes its fingers along panda_hand's Y axis, so a consumer must"
    " post-multiply by Rz(90 deg) to obtain a panda_hand pose."
)

REVOLUTION_PIECES = {"pawn", "rook", "bishop", "queen", "king"}
"""Pieces that are solids of revolution, so grasp azimuth about the piece axis is free."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR, help="Directory of baked piece meshes.")
    parser.add_argument("--gripper-config", type=Path, default=DEFAULT_GRIPPER_CONFIG)
    parser.add_argument("--num-grasps", type=int, default=2000, help="Diffusion samples per piece.")
    parser.add_argument("--topk", type=int, default=500, help="Raw grasps kept from the sampler before filtering.")
    parser.add_argument("--num-sample-points", type=int, default=2048)
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum discriminator score.")
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=75.0,
        help="Maximum angle between the approach direction and straight down.",
    )
    parser.add_argument(
        "--preferred-tilt-deg",
        type=float,
        default=25.0,
        help="Tilt up to which a grasp is ranked on score alone; beyond it, score is penalised.",
    )
    parser.add_argument(
        "--tilt-weight",
        type=float,
        default=1.0,
        help="Score penalty applied per 90 deg of tilt beyond --preferred-tilt-deg.",
    )
    parser.add_argument(
        "--board-clearance",
        type=float,
        default=0.004,
        help="Minimum height [m] of any gripper point above the board plane.",
    )
    parser.add_argument(
        "--max-grasp-width",
        type=float,
        default=0.075,
        help="Widest piece cross-section [m] the Franka fingers may have to span (hardware max is 0.08).",
    )
    parser.add_argument("--keep-per-piece", type=int, default=64, help="Feasible grasps stored per piece.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def rotation_z(angle: float) -> np.ndarray:
    return tra.rotation_matrix(angle, [0.0, 0.0, 1.0])


def approach_tilt_deg(matrix: np.ndarray) -> float:
    """Angle [deg] between the grasp approach (+Z) and straight down. 0 = top-down."""
    approach = matrix[:3, 2]
    return float(np.degrees(np.arccos(np.clip(-approach[2] / np.linalg.norm(approach), -1.0, 1.0))))


def canonical_azimuth(matrix: np.ndarray) -> float:
    """Azimuth [rad] of the grasp about the piece axis.

    Uses the approach direction's horizontal component; for a perfectly vertical
    approach that is degenerate, so fall back to the finger-closing axis.
    """
    approach = matrix[:3, 2]
    if np.linalg.norm(approach[:2]) > 1e-4:
        return float(np.arctan2(approach[1], approach[0]))
    closing = matrix[:3, 0]
    return float(np.arctan2(closing[1], closing[0]))


def gripper_lowest_point(matrix: np.ndarray, gripper_points: np.ndarray) -> float:
    """Height [m] of the lowest gripper vertex once placed at ``matrix``."""
    return float((gripper_points @ matrix[:3, :3].T + matrix[:3, 3])[:, 2].min())


def pinched_width(matrix: np.ndarray, piece_points: np.ndarray, depth: float) -> float:
    """Piece width [m] the fingers must span at ``matrix``, or ``inf`` if they close on nothing.

    Transforms the piece into the grasp frame and keeps whatever sits inside the
    volume swept by the closing fingers (thin in Y, spanning the finger length in
    Z); the X extent of that is what the gripper has to open around.
    """
    local = (piece_points - matrix[:3, 3]) @ matrix[:3, :3]
    inside = local[(np.abs(local[:, 1]) <= 0.012) & (local[:, 2] >= 0.45 * depth) & (local[:, 2] <= 1.05 * depth)]
    if len(inside) < 8:
        return float("inf")
    return float(np.abs(inside[:, 0]).max() * 2.0)


def ranking_score(record: dict, args: argparse.Namespace) -> float:
    """Discriminator score, penalised once the approach tilts past the preferred cone.

    GraspGen's own argmax is usually a near-horizontal side grasp of the shaft.
    Those score well in isolation but are the worst choice on a populated board:
    they sweep sideways through neighbouring squares and need the wrist almost
    level. Penalising tilt keeps the ranking on top-down-ish grasps whenever one
    of comparable quality exists.
    """
    excess = max(0.0, record["tilt_deg"] - args.preferred_tilt_deg)
    return record["score"] - args.tilt_weight * excess / 90.0


def grasp_records(
    grasps: np.ndarray,
    scores: np.ndarray,
    gripper_points: np.ndarray,
    piece_points: np.ndarray,
    depth: float,
    args: argparse.Namespace,
    yaw_free: bool,
) -> tuple[list[dict], dict]:
    """Filter raw GraspGen output down to board-executable grasps."""
    stats = {"raw": int(len(grasps)), "below_score": 0, "too_tilted": 0, "hits_board": 0, "bad_width": 0}
    kept: list[dict] = []
    for matrix, score in zip(grasps, scores):
        if score < args.score_threshold:
            stats["below_score"] += 1
            continue
        tilt = approach_tilt_deg(matrix)
        if tilt > args.max_tilt_deg:
            stats["too_tilted"] += 1
            continue
        if gripper_lowest_point(matrix, gripper_points) < args.board_clearance:
            stats["hits_board"] += 1
            continue
        width = pinched_width(matrix, piece_points, depth)
        if width > args.max_grasp_width:
            stats["bad_width"] += 1
            continue

        azimuth = canonical_azimuth(matrix)
        canonical = rotation_z(-azimuth) @ matrix if yaw_free else matrix.copy()
        tcp = canonical[:3, 3] + canonical[:3, 2] * depth
        record = {
            "score": float(score),
            "matrix": [float(v) for v in canonical.flatten()],
            "tilt_deg": tilt,
            "source_azimuth_deg": float(np.degrees(azimuth)),
            "tcp_height": float(tcp[2]),
            "tcp_radius": float(np.linalg.norm(tcp[:2])),
            "grasp_width": width,
            "clearance": gripper_lowest_point(canonical, gripper_points),
        }
        record["rank_score"] = ranking_score(record, args)
        kept.append(record)
    stats["feasible"] = len(kept)
    kept.sort(key=lambda record: record["rank_score"], reverse=True)
    return kept[: args.keep_per_piece], stats


def plot_piece_grasps(mesh: trimesh.Trimesh, records: list[dict], control_points: np.ndarray, out_path: Path) -> None:
    """Side/top view of the piece with the highest-scoring gripper poses drawn on."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = records[:12]
    figure, (side, top) = plt.subplots(1, 2, figsize=(11, 5))
    points = np.asarray(mesh.sample(6000))
    side.scatter(points[:, 0], points[:, 2], s=0.6, c="0.75", linewidths=0)
    top.scatter(points[:, 0], points[:, 1], s=0.6, c="0.75", linewidths=0)

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, max(len(shown), 1)))
    for record, color in zip(shown, colors):
        matrix = np.asarray(record["matrix"], dtype=float).reshape(4, 4)
        drawn = control_points @ matrix[:3, :3].T + matrix[:3, 3]
        side.plot(drawn[:, 0], drawn[:, 2], color=color, linewidth=1.1, alpha=0.85)
        top.plot(drawn[:, 0], drawn[:, 1], color=color, linewidth=1.1, alpha=0.85)

    side.axhline(0.0, color="saddlebrown", linewidth=2)
    side.set(xlabel="x [m]", ylabel="z [m]", title=f"{out_path.stem} - side view ({len(shown)} best grasps)")
    top.set(xlabel="x [m]", ylabel="y [m]", title="top view")
    for axis in (side, top):
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_path, dpi=130)
    plt.close(figure)


def plot_summary(
    pieces: dict[str, dict],
    meshes: dict[str, trimesh.Trimesh],
    control_points: np.ndarray,
    depth: float,
    out_path: Path,
) -> None:
    """One panel per piece showing the grasp the ranking picked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(pieces), figsize=(3.1 * len(pieces), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (piece, result) in zip(axes, pieces.items()):
        points = np.asarray(meshes[piece].sample(9000))
        axis.scatter(points[:, 0], points[:, 2], s=0.5, c="0.78", linewidths=0)
        best = result["best"]
        matrix = np.asarray(best["matrix"], dtype=float).reshape(4, 4)
        drawn = control_points @ matrix[:3, :3].T + matrix[:3, 3]
        axis.plot(drawn[:, 0], drawn[:, 2], color="#1f77b4", linewidth=2.2)
        tcp = matrix[:3, 3] + matrix[:3, 2] * depth
        axis.plot(tcp[0], tcp[2], marker="o", color="#d62728", markersize=5)
        axis.axhline(0.0, color="saddlebrown", linewidth=2)
        axis.set_title(
            f"{piece}\nscore {best['score']:.2f} | tilt {best['tilt_deg']:.0f}°\n"
            f"grip {best['tcp_height'] * 1000:.0f} mm | span {best['grasp_width'] * 1000:.0f} mm",
            fontsize=9,
        )
        axis.set_xlabel("x [m]")
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("z [m]")
    figure.suptitle("GraspGen (franka_panda): selected grasp per chess piece", fontsize=12)
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    import torch

    torch.manual_seed(args.seed)

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.robot import get_gripper_info

    metadata = json.loads((args.asset_dir / "pieces.json").read_text())
    gripper_info = get_gripper_info(GRIPPER_NAME)
    gripper_points = np.asarray(gripper_info.collision_mesh.vertices)
    control_points = np.asarray(gripper_info.control_points_visualization[0])
    depth = float(gripper_info.depth)
    print(f"[INFO] Gripper '{GRIPPER_NAME}': depth={depth * 1000:.1f} mm, {len(gripper_points)} collision vertices")

    grasp_cfg = load_grasp_cfg(str(args.gripper_config))
    sampler = GraspGenSampler(grasp_cfg)

    out_dir = args.asset_dir / "grasps"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    meshes: dict[str, trimesh.Trimesh] = {}
    for piece, info in metadata["pieces"].items():
        mesh = meshes[piece] = trimesh.load(args.asset_dir / info["obj"], process=False)
        points = np.asarray(trimesh.sample.sample_surface(mesh, args.num_sample_points)[0])
        dense_points = np.asarray(trimesh.sample.sample_surface(mesh, 20000)[0])

        # GraspGen expects a zero-mean point cloud; undo the shift afterwards so the
        # grasps come back in the piece frame.
        centre = points.mean(axis=0)
        grasps_t, scores_t = GraspGenSampler.run_inference(
            points - centre,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk,
            remove_outliers=False,
        )
        if len(grasps_t) == 0:
            print(f"[WARN] no grasps predicted for {piece}")
            continue
        grasps = grasps_t.cpu().numpy().astype(np.float64)
        grasps[:, :3, 3] += centre
        scores = scores_t.cpu().numpy().astype(np.float64)

        yaw_free = piece in REVOLUTION_PIECES
        kept, stats = grasp_records(grasps, scores, gripper_points, dense_points, depth, args, yaw_free)
        if not kept:
            print(f"[WARN] {piece}: no feasible grasp survived filtering ({stats})")
            continue

        best = kept[0]
        height = info["height"]
        results[piece] = {
            "yaw_free": yaw_free,
            "piece_height": height,
            "stats": stats,
            "best": best,
            "grasps": kept,
        }
        print(
            f"[{piece:>6}] {stats['feasible']:4d}/{stats['raw']:4d} feasible "
            f"(score: {stats['below_score']}, tilt: {stats['too_tilted']}, board: {stats['hits_board']},"
            f" width: {stats['bad_width']}) | best score={best['score']:.3f} tilt={best['tilt_deg']:4.1f}deg"
            f" grip_height={best['tcp_height'] * 1000:5.1f} mm ({100 * best['tcp_height'] / height:4.1f}%)"
            f" width={best['grasp_width'] * 1000:4.1f} mm"
        )

        if not args.no_plots:
            plot_piece_grasps(mesh, kept, control_points, out_dir / f"{piece}_grasps.png")

    payload = {
        "gripper": GRIPPER_NAME,
        "gripper_frame": "graspgen",
        "gripper_depth": depth,
        "convention": CONVENTION_NOTE,
        "piece_scale": metadata["scale"],
        "filters": {
            "score_threshold": args.score_threshold,
            "max_tilt_deg": args.max_tilt_deg,
            "board_clearance": args.board_clearance,
            "max_grasp_width": args.max_grasp_width,
        },
        "ranking": {
            "rule": "rank_score = score - tilt_weight * max(0, tilt_deg - preferred_tilt_deg) / 90",
            "preferred_tilt_deg": args.preferred_tilt_deg,
            "tilt_weight": args.tilt_weight,
        },
        "pieces": results,
    }
    out_file = out_dir / "chess_grasps.json"
    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[INFO] Wrote grasps for {len(results)} pieces to {out_file}")

    if not args.no_plots and results:
        summary_path = out_dir / "chess_grasps_summary.png"
        plot_summary(results, meshes, control_points, depth, summary_path)
        print(f"[INFO] Wrote grasp summary figure to {summary_path}")


if __name__ == "__main__":
    main()
