"""Chess board and piece geometry for a Newton ``ModelBuilder``.

The baked assets under ``assets/chess/generated/s150`` already contain everything the
simulator needs: a 200k-triangle render mesh plus 16 CoACD convex hulls that preserve the
neck of each piece, and ``pieces.json`` with the authored mass and dimensions.  This module
reads them once and shares the resulting ``newton.Mesh`` objects across every instance.

Three decisions here are load-bearing and were measured, not guessed:

* ``ModelBuilder.add_usd()`` per piece works, but rebuilds a fresh mesh (and GPU BVH) for
  every shape of every instance: 3118 ms to build the 4x4 scene versus 262 ms going through
  the cache below, and 802 ms versus 231 ms to finalize.
* Colliders must be added with ``add_shape_convex_hull`` (``GeoType.CONVEX_MESH``), never
  ``add_shape_mesh`` (``GeoType.MESH``).  Same geometry, 5.2 ms/frame versus 145.8 ms/frame
  on the 4x4 scene, and the ``MESH`` version is not merely slow -- it explodes.
* The board is a thin static box.  The shipped board assets are zero-thickness quads; as a
  mesh collider the pieces fall through them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Sequence

# openusd 25.11's UsdPhysics parser races and segfaults nondeterministically when driven
# from several threads, which is exactly what Usd.Stage.Open does under the hood. Measured:
# a bare pxr loop over one piece USD died after 5-23 iterations, three runs out of three.
# This must be set before the first pxr import in the process, so it lives at module top.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

# Imported ahead of newton on purpose: importing board_layout strips the repo root from
# sys.path, where this repo's own newton/ directory would shadow the newton package.
from .board_layout import (
    BOARD_ASSET_DIR,
    BOARD_CENTER,
    BOARD_RENDER_LIFT,
    BOARD_THICKNESS,
    GENERATED_ASSET_DIR,
    PIECE_COLORS,
    PIECE_SCALE_TAG,
    TABLE_CENTER,
    TABLE_COLOR,
    TABLE_SIZE,
    TABLE_TOP_Z,
    BoardLayout,
    Vec3,
    board_asset_pose,
)

import newton
import numpy as np
import warp as wp
from pxr import Usd, UsdGeom, UsdShade

PIECE_FRICTION = 1.1
"""Piece friction, matching the lab's ``static_friction=1.1`` rigid body material."""

BOARD_FRICTION = 0.9
TABLE_FRICTION = 0.9

BOARD_FALLBACK_COLOR = (0.55, 0.50, 0.45)
"""Render colour for a board subset with no bound material. The shipped boards all bind
one, so this only shows up if someone points ``board_usd`` at a bare mesh."""

_HULL_PRIM_PREFIX = "/ChessPiece/collisions/hull_"


def _to_transform(position: Sequence[float], quat_xyzw: Sequence[float]) -> wp.transform:
    return wp.transform(wp.vec3(*(float(v) for v in position)), wp.quat(*(float(v) for v in quat_xyzw)))


def _resolve_color(color: str | Sequence[float]) -> wp.vec3:
    """Accept either a piece colour name (``"white"``/``"black"``) or a literal RGB triple."""
    rgb = PIECE_COLORS[color] if isinstance(color, str) else color
    return wp.vec3(float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _mesh_world_points(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points, face vertex counts and indices of a ``UsdGeom.Mesh``, in stage coordinates.

    The baked pieces put a 1.5x scale on the visual prim's xformOp stack, so the local
    transform has to be composed in rather than ignored.
    """
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    return points @ matrix[:3, :3] + matrix[3, :3], counts, indices


def _triangulate(counts: np.ndarray, indices: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Fan-triangulate a polygon soup. Returns ``(M, 3)`` triangles and their source faces.

    Vectorised per polygon size rather than per polygon: the piece visuals have 227 456
    faces each, and a Python loop over them costs ~400 ms per kind (2.4 s for the six).
    The boards ship quads and the pieces ship triangles, so this runs one or two passes.

    The source-face column is what lets ``UsdGeom.Subset`` face indices -- how the boards
    mark their light and dark squares -- survive triangulation.
    """
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    fans, sources = [], []
    for size in np.unique(counts):
        faces_of_size = np.flatnonzero(counts == size)
        corners = indices[starts[faces_of_size][:, None] + np.arange(size)[None, :]]
        for k in range(1, int(size) - 1):
            fans.append(np.stack([corners[:, 0], corners[:, k], corners[:, k + 1]], axis=1))
            sources.append(faces_of_size)
    return np.concatenate(fans) + offset, np.concatenate(sources)


def _diffuse_color(prim: Usd.Prim, fallback: Vec3) -> Vec3:
    """Diffuse colour of the ``UsdPreviewSurface`` bound to a prim.

    Newton builds meshes from raw points and indices, so USD materials are dropped on the
    floor and every render mesh comes out in the viewer's default green unless a colour is
    passed explicitly. The boards bind one material per square colour; reading it back keeps
    the rendered board looking like the authored asset.
    """
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return fallback
    shader = UsdShade.Shader(material.ComputeSurfaceSource()[0])
    if not shader:
        return fallback
    color = shader.GetInput("diffuseColor")
    value = color.Get() if color else None
    return fallback if value is None else (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class PieceInfo:
    """One piece kind's baked metadata, straight out of ``pieces.json``."""

    kind: str
    mass: float
    height: float
    base_diameter: float
    narrowest_width: float
    narrowest_width_height: float
    volume: float
    num_collision_hulls: int


class MassProperties(NamedTuple):
    """Rigid-body mass properties of a piece: centre of mass and inertia tensor about it."""

    com: wp.vec3
    inertia: wp.mat33


class PieceAssets:
    """Lazy, process-wide cache of the baked ``s150`` piece geometry.

    One ``newton.Mesh`` per collision hull and one per visual mesh, per *kind* -- shared by
    every instance of that kind, in every world.  Reading a kind costs ~4 ms for the 16
    hulls and 20-40 ms for the 113k-300k vertex visual; instancing a cached kind then costs
    ~0.3 ms for its 16 colliders plus ~6 ms for the visual shape.

    Do not add a serialised cache on top of this: an ``npz`` of the same data measured
    4.6x slower to reload than re-reading the USD. The baked ``.usd`` files *are* the cache.
    """

    def __init__(self, generated_dir: Path | str = GENERATED_ASSET_DIR, scale_tag: str = PIECE_SCALE_TAG):
        self.dir = Path(generated_dir) / scale_tag
        metadata_path = self.dir / "pieces.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing baked chess assets at {self.dir}. Run:\n"
                f"    python lab/scripts/prepare_chess_assets.py --scale {int(scale_tag[1:]) / 100}"
            )
        metadata = json.loads(metadata_path.read_text())["pieces"]
        self._info = {
            kind: PieceInfo(
                kind=kind,
                mass=float(entry["mass"]),
                height=float(entry["height"]),
                base_diameter=float(entry["base_diameter"]),
                narrowest_width=float(entry["narrowest_width"]),
                narrowest_width_height=float(entry["narrowest_width_height"]),
                volume=float(entry["volume"]),
                num_collision_hulls=int(entry["num_collision_hulls"]),
            )
            for kind, entry in metadata.items()
        }
        self._hulls: dict[str, list[newton.Mesh]] = {}
        self._visual: dict[str, newton.Mesh] = {}
        self._mass_props: dict[str, MassProperties] = {}

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self._info)

    def info(self, kind: str) -> PieceInfo:
        if kind not in self._info:
            raise KeyError(f"No baked asset for piece kind {kind!r}; have {self.kinds}.")
        return self._info[kind]

    def mass(self, kind: str) -> float:
        return self.info(kind).mass

    def height(self, kind: str) -> float:
        return self.info(kind).height

    def base_diameter(self, kind: str) -> float:
        return self.info(kind).base_diameter

    def narrowest_width(self, kind: str) -> float:
        """Width [m] at the piece's neck -- the gap a gripper has to close onto."""
        return self.info(kind).narrowest_width

    def hulls(self, kind: str) -> list[newton.Mesh]:
        """The 16 CoACD collision hulls, 32 vertices each, in the piece's own frame."""
        if kind not in self._hulls:
            stage = Usd.Stage.Open(str(self._usd_path(kind)))
            meshes = []
            for prim in stage.Traverse():
                if not str(prim.GetPath()).startswith(_HULL_PRIM_PREFIX):
                    continue
                mesh = UsdGeom.Mesh(prim)
                vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
                # compute_inertia stays on: 32-vertex hulls make it free, and mass_properties
                # below needs the per-hull volume/com/inertia it produces.
                meshes.append(newton.Mesh(vertices, indices))
            expected = self.info(kind).num_collision_hulls
            if len(meshes) != expected:
                raise ValueError(f"{kind}.usd has {len(meshes)} collision hulls, pieces.json says {expected}.")
            self._hulls[kind] = meshes
        return self._hulls[kind]

    def visual(self, kind: str) -> newton.Mesh:
        """The render mesh, with the asset's 1.5x xformOp already baked into the points."""
        if kind not in self._visual:
            stage = Usd.Stage.Open(str(self._usd_path(kind)))
            vertices: list[np.ndarray] = []
            triangles: list[np.ndarray] = []
            offset = 0
            for prim in stage.Traverse():
                if not prim.IsA(UsdGeom.Mesh) or str(prim.GetPath()).startswith(_HULL_PRIM_PREFIX):
                    continue
                points, counts, indices = _mesh_world_points(prim)
                triangles.append(_triangulate(counts, indices, offset)[0])
                vertices.append(points)
                offset += len(points)
            self._visual[kind] = newton.Mesh(
                np.concatenate(vertices).astype(np.float32),
                np.concatenate(triangles).astype(np.int32).flatten(),
                compute_inertia=False,
            )
        return self._visual[kind]

    def mass_properties(self, kind: str) -> MassProperties:
        """Centre of mass and inertia tensor about it, for the authored mass.

        The hulls are added with ``density=0.0`` (see :meth:`add_piece`), so the body would
        otherwise end up with a zero inertia tensor that Newton silently replaces with an
        isotropic 1e-6 fallback -- a pawn would then be as hard to tip over as it is to
        spin, and its centre of mass would sit at the base instead of 24 mm up.

        Summing the hulls at unit density and rescaling to the authored mass reproduces
        ``add_usd``'s own numbers to float32 precision, and is immune to the ~16% volume
        inflation from overlapping hulls because only the *ratio* survives the rescale.
        """
        if kind not in self._mass_props:
            hulls = self.hulls(kind)
            volume = sum(hull.mass for hull in hulls)
            com = sum(hull.mass * np.asarray(hull.com, dtype=np.float64) for hull in hulls) / volume
            inertia = np.zeros((3, 3))
            for hull in hulls:
                shift = np.asarray(hull.com, dtype=np.float64) - com
                inertia += np.asarray(hull.inertia, dtype=np.float64).reshape(3, 3) + hull.mass * (
                    np.dot(shift, shift) * np.eye(3) - np.outer(shift, shift)
                )
            inertia *= self.mass(kind) / volume
            self._mass_props[kind] = MassProperties(
                wp.vec3(*com.tolist()), wp.mat33(*inertia.flatten().tolist())
            )
        return self._mass_props[kind]

    def add_piece(
        self,
        builder: newton.ModelBuilder,
        kind: str,
        xform: wp.transform,
        color: str | Sequence[float],
        visual: bool = True,
        label: str | None = None,
    ) -> int:
        """Add one piece as a free rigid body. Returns its body index.

        ``add_body`` already creates the ``JointType.FREE`` joint -- calling
        ``add_joint_free`` on top of it gives the body two free joints and breaks the model.

        ``visual=False`` drops the render mesh entirely; the piece is then invisible but
        physically identical, which is what headless runs want (the visual meshes cost
        ~310 ms of the 8x8 scene's finalize and nothing per step).
        """
        props = self.mass_properties(kind)
        body = builder.add_body(
            xform=xform, mass=self.mass(kind), com=props.com, inertia=props.inertia, label=label
        )
        # density=0.0 is required, not an optimisation: the 16 hulls overlap, so letting
        # Newton integrate their volumes gives a pawn 0.029466 kg instead of the authored
        # 0.025362 kg (+16%). Mass and inertia come from add_body instead.
        collider_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, mu=PIECE_FRICTION, restitution=0.0, is_visible=False
        )
        for hull in self.hulls(kind):
            builder.add_shape_convex_hull(body=body, mesh=hull, cfg=collider_cfg)
        if visual:
            builder.add_shape_mesh(
                body=body,
                mesh=self.visual(kind),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, has_shape_collision=False, has_particle_collision=False, is_visible=True
                ),
                color=_resolve_color(color),
                label=None if label is None else f"{label}_visual",
            )
        return body

    def _usd_path(self, kind: str) -> Path:
        self.info(kind)
        return self.dir / f"{kind}.usd"


class BoardSubset(NamedTuple):
    """One ``UsdGeom.Subset`` of the board -- in practice, the light or the dark squares."""

    name: str
    indices: np.ndarray  # (3 * M,) int32, flattened triangles into BoardMesh.vertices
    color: Vec3


class BoardMesh(NamedTuple):
    """A triangulated board render mesh and its asset-local axis-aligned bounds."""

    vertices: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (3 * M,) int32, flattened triangles
    lower: np.ndarray  # (3,) float32
    upper: np.ndarray  # (3,) float32
    subsets: tuple[BoardSubset, ...]


@lru_cache(maxsize=None)
def board_mesh_and_bbox(board_usd: str) -> BoardMesh:
    """Read a board asset, triangulating it on the way in.

    ``newton.Mesh`` wants triangle indices and the shipped boards are all quads
    (``faceVertexCounts`` is uniformly 4), so the fan-triangulation is mandatory, not a
    convenience.  ``board_usd`` may be a bare file name (resolved against
    ``assets/chess/board``) or a full path, so ``layout.board_usd`` and
    ``layout.board_usd_path`` both work.
    """
    path = Path(board_usd)
    if not path.is_absolute():
        path = BOARD_ASSET_DIR / path
    stage = Usd.Stage.Open(str(path))
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    subsets: list[BoardSubset] = []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points, counts, indices = _mesh_world_points(prim)
        tris, sources = _triangulate(counts, indices, offset)
        for child in prim.GetChildren():
            subset = UsdGeom.Subset(child)
            if not subset or subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
                continue
            faces = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64)
            selected = tris[np.isin(sources, faces)]
            subsets.append(
                BoardSubset(
                    child.GetName(),
                    selected.astype(np.int32).flatten(),
                    _diffuse_color(child, BOARD_FALLBACK_COLOR),
                )
            )
        triangles.append(tris)
        vertices.append(points)
        offset += len(points)
    points = np.concatenate(vertices).astype(np.float32)
    return BoardMesh(
        points,
        np.concatenate(triangles).astype(np.int32).flatten(),
        points.min(0),
        points.max(0),
        tuple(subsets),
    )


def add_board(
    builder: newton.ModelBuilder,
    layout: BoardLayout,
    board_center: tuple[float, float] = BOARD_CENTER,
    table_top_z: float = TABLE_TOP_Z,
    visual: bool = True,
) -> int:
    """Add the board: a static thin box collider, plus optional render meshes.

    Returns the collider's shape index.  The box's half-extents come from the asset's real
    bounding box times ``layout.board_scale``, so it fits whichever board the scenario
    picked, and its top face sits exactly at ``table_top_z``.

    The render meshes are scaled ``(s, s, 1.0)`` -- the same anisotropic scale the lab
    applies -- because ``board_scale`` stretches the squares, not the (zero) board
    thickness.  One mesh per ``UsdGeom.Subset`` so the light and dark squares keep their
    authored colours; boards with no subsets get a single mesh.
    """
    board = board_mesh_and_bbox(layout.board_usd)
    scale = layout.board_scale
    position, quat = board_asset_pose(layout, board_center, table_top_z)
    asset_xform = _to_transform(position, quat)

    center_x = float((board.lower[0] + board.upper[0]) / 2.0 * scale)
    center_y = float((board.lower[1] + board.upper[1]) / 2.0 * scale)
    box_center = wp.transform_point(asset_xform, wp.vec3(center_x, center_y, -BOARD_THICKNESS / 2.0))
    collider = builder.add_shape_box(
        body=-1,
        xform=wp.transform(box_center, wp.quat(*quat)),
        hx=float((board.upper[0] - board.lower[0]) / 2.0 * scale),
        hy=float((board.upper[1] - board.lower[1]) / 2.0 * scale),
        hz=BOARD_THICKNESS / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(mu=BOARD_FRICTION, restitution=0.0, is_visible=False),
        label="board_collider",
    )
    if visual:
        render_xform = _to_transform((position[0], position[1], position[2] + BOARD_RENDER_LIFT), quat)
        parts = board.subsets or (BoardSubset("squares", board.indices, BOARD_FALLBACK_COLOR),)
        for part in parts:
            builder.add_shape_mesh(
                body=-1,
                xform=render_xform,
                mesh=newton.Mesh(board.vertices, part.indices, compute_inertia=False),
                scale=wp.vec3(scale, scale, 1.0),
                cfg=newton.ModelBuilder.ShapeConfig(
                    has_shape_collision=False, has_particle_collision=False, is_visible=True
                ),
                color=wp.vec3(*part.color),
                label=f"board_visual_{part.name}",
            )
    return collider


def add_table(
    builder: newton.ModelBuilder,
    center: tuple[float, float] = TABLE_CENTER,
    size: Vec3 = TABLE_SIZE,
    table_top_z: float = TABLE_TOP_Z,
    color: Sequence[float] = TABLE_COLOR,
    visual: bool = True,
) -> int:
    """Add the lab's static 1.30 x 1.10 x 0.77 m table, its top face at ``table_top_z``."""
    return builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(center[0], center[1], table_top_z - size[2] / 2.0), wp.quat_identity()),
        hx=size[0] / 2.0,
        hy=size[1] / 2.0,
        hz=size[2] / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(mu=TABLE_FRICTION, restitution=0.0, is_visible=visual),
        color=wp.vec3(float(color[0]), float(color[1]), float(color[2])),
        label="table",
    )


def contact_budget(n_pieces: int) -> int:
    """Rigid contact capacity for a scene with this many pieces, summed over all worlds.

    Undersizing this is a *silent* total failure, not a warning: the collision pipeline
    drops every contact and each piece free-falls (dz = -311.67 mm after 60 frames at
    dt=1/240, i.e. exactly 0.5*g*t^2). Measured peaks are ~1450 contacts per piece --
    22 732 for the 16-piece 4x4 scene, 44 944 for the 32-piece 8x8 -- so budget 1600 each
    and round up to a power of two, with a floor that covers the small scenarios' board and
    table contacts too.
    """
    needed = int(1600 * n_pieces)
    return 1 << max(15, (needed - 1).bit_length())
