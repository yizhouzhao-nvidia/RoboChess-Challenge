"""Bake grasp-ready chess piece assets for the Franka picking task.

The shipped ``assets/chess/*.usdc`` pieces are render-only: a single 200k-triangle
mesh with no physics schemas.  To pick them up we need (a) a rigid body with a
collision approximation that keeps the *neck* of the piece -- a convex hull would
fill it in and there would be nothing to grip -- and (b) the exact same geometry
as a mesh file so GraspGen predicts grasps for what the simulator actually
simulates.

This script produces both, at a user-chosen scale:

    assets/chess/generated/s<scale>/<piece>.usd   rigid body + CoACD collision hulls
    assets/chess/generated/s<scale>/<piece>.obj   the same geometry, for GraspGen
    assets/chess/generated/s<scale>/pieces.json   dimensions / mass / metadata

It only needs ``pxr``, ``trimesh`` and ``coacd`` -- no Isaac Sim app -- so it runs
in a couple of minutes:

.. code-block:: bash

    python lab/scripts/prepare_chess_assets.py --scale 1.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

REPO_ROOT = Path(__file__).resolve().parents[2]
CHESS_ASSET_DIR = REPO_ROOT / "assets" / "chess"

PIECE_NAMES = ("pawn", "rook", "knight", "bishop", "queen", "king")
"""Chess pieces shipped as ``assets/chess/<name>.usdc``."""

DEFAULT_DENSITY = 900.0
"""Density [kg/m^3] used to derive piece mass (roughly hardwood / dense plastic)."""

MIN_MASS = 0.010
"""Lower bound on piece mass [kg]; PhysX solves grasps of very light bodies poorly."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Uniform scale baked into the pieces. 1.0 keeps the shipped ~47 mm pawn.",
    )
    parser.add_argument("--pieces", nargs="+", default=list(PIECE_NAMES), choices=PIECE_NAMES)
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY, help="Piece density [kg/m^3].")
    parser.add_argument(
        "--coacd-threshold",
        type=float,
        default=0.04,
        help="CoACD concavity threshold. Lower keeps the neck sharper but adds hulls.",
    )
    parser.add_argument("--coacd-max-hulls", type=int, default=16, help="Maximum convex hulls per piece.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to assets/chess/generated/s<scale*100>.",
    )
    return parser.parse_args()


def output_dir_for_scale(scale: float) -> Path:
    """Scale-tagged output directory, so a stale bake can never be loaded by mistake."""
    return CHESS_ASSET_DIR / "generated" / f"s{round(scale * 100):03d}"


def load_piece_mesh(usd_path: Path) -> trimesh.Trimesh:
    """Concatenate every mesh in a chess piece USD into one world-space trimesh."""
    stage = Usd.Stage.Open(str(usd_path))
    parts: list[trimesh.Trimesh] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if counts.size and not np.all(counts == 3):
            raise RuntimeError(f"{usd_path.name}: expected a triangulated mesh, got face sizes {np.unique(counts)}")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        part = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        part.apply_transform(transform)
        parts.append(part)
    if not parts:
        raise RuntimeError(f"No UsdGeom.Mesh found in {usd_path}")
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def normalize_piece(mesh: trimesh.Trimesh, scale: float) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Scale the piece and move its frame to (axis centre, base plane).

    Both the simulator and GraspGen then agree on the piece frame: +Z is up along
    the piece axis, z=0 is the table contact plane, and the origin sits on the
    board square centre.

    Returns the normalised mesh and the recentring translation *in source units*,
    which the USD visual reference re-applies before its scale op.
    """
    mesh = mesh.copy()
    mesh.process(validate=True)
    lower, upper = mesh.bounds
    recenter = np.array([-(lower[0] + upper[0]) / 2.0, -(lower[1] + upper[1]) / 2.0, -lower[2]])
    mesh.apply_translation(recenter)
    mesh.apply_scale(scale)
    return mesh, recenter


def convex_parts(mesh: trimesh.Trimesh, threshold: float, max_hulls: int) -> list[trimesh.Trimesh]:
    """Approximate convex decomposition, so the gripper can reach into the neck."""
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=threshold,
        max_convex_hull=max_hulls,
        preprocess_mode="auto",
        merge=True,
        decimate=True,
        max_ch_vertex=32,
    )
    return [trimesh.Trimesh(vertices=np.asarray(v), faces=np.asarray(f), process=False) for v, f in parts]


def neck_profile(mesh: trimesh.Trimesh, num_slices: int = 60) -> tuple[float, float]:
    """Return (narrowest graspable width [m], its height above the base [m]).

    Sweeps horizontal slices and reports the thinnest one in the upper 75% of the
    piece -- the shaft/neck a parallel-jaw gripper wants.  Purely informational,
    but it is the number that tells you whether a scale is pickable at all.
    """
    height = float(mesh.bounds[1][2])
    heights = np.linspace(0.25 * height, 0.95 * height, num_slices)
    best_width, best_z = float("inf"), 0.0
    for z in heights:
        section = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
        if section is None:
            continue
        planar = np.asarray(section.vertices)[:, :2]
        width = float(np.linalg.norm(planar, axis=1).max() * 2.0)
        if width < best_width:
            best_width, best_z = width, float(z)
    return best_width, best_z


def author_piece_usd(
    out_path: Path,
    visual_usd_relpath: str,
    scale: float,
    hulls: list[trimesh.Trimesh],
    mass: float,
    recenter: np.ndarray,
) -> None:
    """Write a rigid-body USD: referenced visual mesh + explicit convex collision hulls."""
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, Sdf.Path("/ChessPiece"))
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(float(mass))

    # Visual: reference the shipped high-resolution mesh, then apply the same
    # normalisation (recentre, then scale) that the collision hulls were baked with.
    # USD applies xformOps left-to-right on row vectors, so translate-then-scale
    # matches ``normalize_piece``: p' = (p + recenter) * scale.
    visual = UsdGeom.Xform.Define(stage, Sdf.Path("/ChessPiece/visual"))
    visual.GetPrim().GetReferences().AddReference(visual_usd_relpath)
    visual.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in recenter]))
    visual.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    collisions = UsdGeom.Xform.Define(stage, Sdf.Path("/ChessPiece/collisions"))
    UsdGeom.Imageable(collisions).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    for index, hull in enumerate(hulls):
        prim_path = Sdf.Path(f"/ChessPiece/collisions/hull_{index:02d}")
        hull_mesh = UsdGeom.Mesh.Define(stage, prim_path)
        hull_mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in hull.vertices]))
        hull_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(hull.faces)))
        hull_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(hull.faces.astype(np.int32).flatten().tolist()))
        hull_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdPhysics.CollisionAPI.Apply(hull_mesh.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(hull_mesh.GetPrim())
        mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)

    stage.GetRootLayer().Save()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or output_dir_for_scale(args.scale)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict] = {}
    for piece in args.pieces:
        source_usd = CHESS_ASSET_DIR / f"{piece}.usdc"
        mesh, recenter = normalize_piece(load_piece_mesh(source_usd), args.scale)
        hulls = convex_parts(mesh, args.coacd_threshold, args.coacd_max_hulls)
        mass = max(MIN_MASS, float(mesh.volume) * args.density)
        width, width_z = neck_profile(mesh)

        author_piece_usd(
            out_path=out_dir / f"{piece}.usd",
            visual_usd_relpath=f"../../{piece}.usdc",
            scale=args.scale,
            hulls=hulls,
            mass=mass,
            recenter=recenter,
        )
        mesh.export(out_dir / f"{piece}.obj")

        extents = mesh.extents
        metadata[piece] = {
            "scale": args.scale,
            "height": float(extents[2]),
            "base_diameter": float(max(extents[0], extents[1])),
            "narrowest_width": float(width),
            "narrowest_width_height": float(width_z),
            "mass": float(mass),
            "volume": float(mesh.volume),
            "num_collision_hulls": len(hulls),
            "usd": f"{piece}.usd",
            "obj": f"{piece}.obj",
        }
        print(
            f"[{piece:>6}] height={extents[2] * 1000:5.1f} mm  base={max(extents[:2]) * 1000:5.1f} mm  "
            f"neck={width * 1000:4.1f} mm @ {width_z * 1000:4.1f} mm  mass={mass * 1000:5.1f} g  "
            f"hulls={len(hulls)}"
        )

    (out_dir / "pieces.json").write_text(json.dumps({"scale": args.scale, "pieces": metadata}, indent=2) + "\n")
    print(f"\n[INFO] Wrote {len(metadata)} pieces to {out_dir}")


if __name__ == "__main__":
    main()
