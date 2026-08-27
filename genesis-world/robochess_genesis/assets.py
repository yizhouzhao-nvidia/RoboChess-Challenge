"""Chess piece, board and table geometry, translated from USD into what Genesis can load.

Genesis loads geometry from files: ``.obj/.stl/.dae/.glb/.gltf`` meshes, URDF, MJCF and
(partially) USD.  It has no in-memory mesh handle to hand a solver, which is the whole
interface the Newton port used -- ``newton.Mesh`` objects built from raw numpy and shared
across every instance of a piece kind.  So this module is the piece of the port with no
counterpart on the Newton side: it **transcodes** the baked assets once into a disk cache
and hands Genesis paths.

What comes out of the cache, per piece kind:

* ``<kind>_hull_NN.obj`` -- the 16 CoACD collision hulls, 32 vertices each, exactly the
  geometry ``assets/chess/generated/s150/<kind>.usd`` carries;
* ``<kind>_visual.obj`` -- the render mesh, with the asset's 1.5x xformOp baked into the
  points, decimated (see :data:`DEFAULT_VISUAL_FACES`);
* ``<kind>.urdf`` -- a one-link URDF binding those together: 16 ``<collision>`` geoms, one
  ``<visual>``, and an ``<inertial>`` carrying the authored mass with the inertia tensor
  computed here.

**Why a URDF and not sixteen mesh entities.**  A chess piece is one rigid body with 16
colliders.  ``gs.morphs.Mesh`` is one file per entity and ``gs.morphs.MeshSet`` gives no
way to mark a submesh visual-only, so a URDF is the only Genesis morph that expresses
"one body, many collision geoms, one visual geom, this inertia".  It is also what makes
the piece's mass properties *exact* rather than integrated from overlapping hulls -- see
:meth:`PieceAssets.mass_properties`.

**Why the visual mesh is decimated and the Newton port's is not.**  Newton builds one
``newton.Mesh`` per kind and instances it, so a 227k-face piece costs one upload.  Genesis
builds the mesh per *entity*: a 32-piece 8x8 board is 32 separate copies, 7.3 M triangles,
and both the loader and the rasteriser feel it.  The colliders are untouched -- the physics
is bit-identical either way -- and ``--visual-faces 0`` restores the full-resolution
render mesh for anyone who wants a beauty render.

The cache lives in ``$ROBOCHESS_GENESIS_CACHE`` (default ``~/.cache/robochess-genesis``)
and is keyed on the source files' size and mtime plus the transcoding parameters, so
re-baking the assets or changing ``--visual-faces`` regenerates rather than silently
serving the old geometry.  Building all six kinds cold takes ~20 s; a warm cache is a
handful of ``stat`` calls.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Sequence
from xml.sax.saxutils import escape

# openusd's Usd.Stage parser races and segfaults nondeterministically when driven from
# several threads. This must be set before the first pxr import in the process, and
# board_layout does exactly that -- import it before pxr, not for what it exports.
from .board_layout import (
    BOARD_ASSET_DIR,
    BOARD_CENTER,
    BOARD_RENDER_LIFT,
    BOARD_THICKNESS,
    GENERATED_ASSET_DIR,
    PIECE_COLORS,
    PIECE_SCALE_TAG,
    TABLE_CENTER,
    TABLE_SIZE,
    TABLE_TOP_Z,
    BoardLayout,
    Vec3,
    board_asset_pose,
)

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade

__all__ = [
    "BOARD_FRICTION",
    "DEFAULT_VISUAL_FACES",
    "PIECE_FRICTION",
    "TABLE_FRICTION",
    "BoardMesh",
    "BoardSubset",
    "MassProperties",
    "PieceAssets",
    "PieceInfo",
    "board_geometry",
    "cache_root",
    "collision_pair_budget",
]

PIECE_FRICTION = 1.1
"""Piece friction, matching the lab's ``static_friction=1.1`` rigid body material.

Genesis, like Newton, takes a single coefficient where Isaac Lab has a static and a
dynamic one; the pieces keep the static value and the surfaces the dynamic one, which is
the same split the Newton port documents."""

BOARD_FRICTION = 0.9
TABLE_FRICTION = 0.9

BOARD_FALLBACK_COLOR = (0.55, 0.50, 0.45)
"""Render colour for a board subset with no bound material. The shipped boards all bind
one, so this only shows up if someone points ``board_usd`` at a bare mesh."""

DEFAULT_VISUAL_FACES = 6000
"""Target triangle count for a decimated piece render mesh.

The baked pieces carry 227 456 faces each -- authored for an offline render, not for 32
independent copies in a rasteriser.  At 6 000 a 70 mm piece filling ~90 px of a 1280x720
frame is visually indistinguishable from the original (checked on rendered PNGs of the
king and the knight, the two with real silhouette detail).  0 disables decimation."""

_HULL_PRIM_PREFIX = "/ChessPiece/collisions/hull_"

_CACHE_VERSION = 4
"""Bumped whenever the transcoder's *output* changes meaning, so an old cache is not
served to new code even if the inputs are untouched."""


def cache_root() -> Path:
    """Where transcoded geometry is written. ``$ROBOCHESS_GENESIS_CACHE`` overrides."""
    override = os.environ.get("ROBOCHESS_GENESIS_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "robochess-genesis"


def _referenced_layers(path: Path) -> list[Path]:
    """Every external layer *path* composes in, resolved to absolute paths.

    The baked piece assets are an 11 kB wrapper that carries the collision hulls inline and
    *references* the 10 MB render mesh next door (``generated/s150/pawn.usd`` ->
    ``assets/chess/pawn.usdc``).  Stamping only the wrapper would mean re-baking or editing
    that payload left the wrapper's mtime untouched and the cache happily served the old
    visual mesh -- a stale render with no symptom anywhere else.

    ``Sdf.Layer.FindOrOpen`` reads the wrapper without composing the stage, so this stays a
    small read rather than pulling the payload in.  Non-recursive on purpose: this repo's
    assets are one level deep, and the wrapper is the only file that references anything.
    """
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        return []
    return [Path(layer.ComputeAbsolutePath(dep)) for dep in layer.GetCompositionAssetDependencies()]


def _stamp(*paths: Path) -> list[list]:
    """Cheap identity of the input files -- path, size and mtime -- including what they
    reference.

    Not a content hash on purpose.  The referenced ``pawn.usdc`` alone is 10 MB and there
    are six of them, so hashing would put ~0.3 s of I/O on every start to protect against a
    case -- a file edited in place to the same size within the same mtime nanosecond -- that
    does not occur.  A ``git checkout`` bumps mtime and regenerates, which is the
    conservative direction.
    """
    stamps = []
    for path in paths:
        for target in (path, *_referenced_layers(path)):
            try:
                stat = target.stat()
            except OSError:
                # A dangling reference is the asset's problem, not the cache's; record it so
                # the key still changes if it later appears.
                stamps.append([str(target), None, None])
                continue
            stamps.append([str(target), stat.st_size, stat.st_mtime_ns])
    return stamps


def _cached_dir(name: str, key: dict) -> tuple[Path, bool]:
    """``(directory, is_fresh)`` for a cache entry described by *key*.

    A stale entry is deleted rather than written over: a half-written previous run would
    otherwise leave some files new and some old, which is the one failure mode that would
    not announce itself.
    """
    directory = cache_root() / name
    manifest = directory / "manifest.json"
    payload = {"version": _CACHE_VERSION, **key}
    if manifest.exists():
        try:
            if json.loads(manifest.read_text()) == payload:
                return directory, True
        except (json.JSONDecodeError, OSError):
            pass
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory, False


def _write_manifest(directory: Path, key: dict) -> None:
    """Written last, so a crash mid-transcode leaves the entry stale rather than wrong."""
    (directory / "manifest.json").write_text(json.dumps({"version": _CACHE_VERSION, **key}, indent=1))


##
# USD reading. Same helpers as the Newton port's assets.py, minus the newton.Mesh calls.
##


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
    faces each, and a Python loop over them costs ~400 ms per kind.

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

    The boards bind one material per square colour; reading it back keeps the rendered
    board looking like the authored asset instead of Genesis' default grey.
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


def _as_trimesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """A ``Trimesh`` with a positive volume.

    CoACD writes its hulls with consistent winding, but a mesh that came back inside-out
    would give a *negative* volume and therefore a negative mass in
    :meth:`PieceAssets.mass_properties` -- a wrong answer that no later stage checks.
    Flipping is cheaper than trusting.
    """
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces), process=False)
    if mesh.volume < 0.0:
        mesh.invert()
    return mesh


def _decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric-decimate to about *target_faces* triangles; a no-op if already smaller."""
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=target_faces)


##
# Piece metadata and mass properties
##


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

    com: np.ndarray  # (3,)
    inertia: np.ndarray  # (3, 3), about the centre of mass


class PieceAssets:
    """Transcoded chess-piece geometry, cached on disk and shared across every instance.

    Construction reads ``pieces.json`` only.  The USD work happens on the first
    :meth:`urdf_path` for a kind and is then a file-existence check.
    """

    def __init__(
        self,
        generated_dir: Path | str = GENERATED_ASSET_DIR,
        scale_tag: str = PIECE_SCALE_TAG,
        visual_faces: int = DEFAULT_VISUAL_FACES,
    ) -> None:
        self.dir = Path(generated_dir) / scale_tag
        self.scale_tag = scale_tag
        self.visual_faces = int(visual_faces)
        metadata_path = self.dir / "pieces.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing baked chess assets at {self.dir}. They are committed through git-lfs:\n"
                f"    git lfs pull\n"
                f"or re-bake them with:\n"
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
        self._mass_props: dict[str, MassProperties] = {}
        self._urdf: dict[str, Path] = {}

    ##
    # Metadata
    ##

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

    def _usd_path(self, kind: str) -> Path:
        self.info(kind)
        return self.dir / f"{kind}.usd"

    ##
    # Transcoding
    ##

    def _read_usd(self, kind: str) -> tuple[list[trimesh.Trimesh], trimesh.Trimesh]:
        """``(hulls, visual)`` straight out of the baked USD, in the piece's own frame."""
        stage = Usd.Stage.Open(str(self._usd_path(kind)))
        hulls: list[trimesh.Trimesh] = []
        visual_vertices: list[np.ndarray] = []
        visual_faces: list[np.ndarray] = []
        offset = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if str(prim.GetPath()).startswith(_HULL_PRIM_PREFIX):
                mesh = UsdGeom.Mesh(prim)
                vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
                hulls.append(_as_trimesh(vertices, indices.reshape(-1, 3)))
                continue
            points, counts, indices = _mesh_world_points(prim)
            visual_faces.append(_triangulate(counts, indices, offset)[0])
            visual_vertices.append(points)
            offset += len(points)

        expected = self.info(kind).num_collision_hulls
        if len(hulls) != expected:
            raise ValueError(f"{kind}.usd has {len(hulls)} collision hulls, pieces.json says {expected}.")
        if not visual_vertices:
            raise ValueError(f"{kind}.usd has no visual mesh outside {_HULL_PRIM_PREFIX}*.")
        visual = trimesh.Trimesh(
            vertices=np.concatenate(visual_vertices), faces=np.concatenate(visual_faces), process=False
        )
        return hulls, visual

    def mass_properties(self, kind: str) -> MassProperties:
        """Centre of mass and inertia tensor about it, for the authored mass.

        Summing the hulls at unit density and rescaling to the authored mass is what the
        Newton port does, and for the same two reasons.  The hulls *overlap*, so letting
        anything integrate their volume directly inflates a pawn's mass by 16 % (0.029466
        kg against the authored 0.025362); and only the *ratio* of the summed tensor to the
        summed volume survives the rescale, so the overlap cancels out of the answer.

        The Genesis-specific part is that this number has to be written into the URDF's
        ``<inertial>``.  Genesis would otherwise recompute the inertia from the collision
        geometry (``recompute_inertia``) or, worse, take the union of 16 overlapping hulls
        at the URDF's density -- the same 16 % error, plus a centre of mass at the base
        instead of 24 mm up, which is what decides whether a pawn tips when a finger
        grazes it.
        """
        if kind not in self._mass_props:
            hulls, _ = self._read_usd(kind)
            volume = sum(hull.volume for hull in hulls)
            com = sum(hull.volume * np.asarray(hull.center_mass) for hull in hulls) / volume
            inertia = np.zeros((3, 3))
            for hull in hulls:
                shift = np.asarray(hull.center_mass) - com
                # trimesh reports moment_inertia about the mesh's own centre of mass at
                # unit density, so this is the parallel-axis shift onto the common centre.
                inertia += np.asarray(hull.moment_inertia) + hull.volume * (
                    np.dot(shift, shift) * np.eye(3) - np.outer(shift, shift)
                )
            inertia *= self.mass(kind) / volume
            self._mass_props[kind] = MassProperties(com=com, inertia=inertia)
        return self._mass_props[kind]

    def urdf_path(self, kind: str) -> Path:
        """Path to this kind's generated one-link URDF, transcoding it if need be."""
        if kind in self._urdf:
            return self._urdf[kind]

        source = self._usd_path(kind)
        key = {
            "kind": kind,
            "scale_tag": self.scale_tag,
            "visual_faces": self.visual_faces,
            "sources": _stamp(source),
        }
        directory, fresh = _cached_dir(f"pieces/{self.scale_tag}/{kind}", key)
        urdf = directory / f"{kind}.urdf"
        if not fresh or not urdf.exists():
            self._transcode(kind, directory)
            _write_manifest(directory, key)
        self._urdf[kind] = urdf
        return urdf

    def _transcode(self, kind: str, directory: Path) -> None:
        hulls, visual = self._read_usd(kind)
        for index, hull in enumerate(hulls):
            hull.export(directory / f"{kind}_hull_{index:02d}.obj")
        _decimate(visual, self.visual_faces).export(directory / f"{kind}_visual.obj")
        (directory / f"{kind}.urdf").write_text(self._urdf_text(kind, len(hulls)))

    def _urdf_text(self, kind: str, hull_count: int) -> str:
        """A one-link URDF: the authored mass and inertia, 16 colliders, one visual.

        Mesh filenames are bare and relative, which is the one form both of Genesis'
        URDF parsers resolve the same way (against the URDF's own directory).
        ``package://`` is not used: the primary parser resolves it by string-stripping the
        scheme and joining with the URDF directory, which is not where a ROS package root
        would be, and that mismatch is exactly what makes Genesis fall back to its legacy
        parser -- see :func:`~robochess_genesis.robots.stage_asset`.
        """
        props = self.mass_properties(kind)
        com = props.com
        inertia = props.inertia
        collisions = "\n".join(
            f'    <collision>\n'
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
            f'      <geometry><mesh filename="{escape(f"{kind}_hull_{index:02d}.obj")}"/></geometry>\n'
            f'    </collision>'
            for index in range(hull_count)
        )
        return f"""<?xml version="1.0"?>
<!-- Generated by robochess_genesis.assets from assets/chess/generated/{self.scale_tag}/{kind}.usd.
     Do not edit: it is regenerated whenever the source USD or the visual-faces target
     changes. Note for future edits: an XML comment may not contain a double hyphen. -->
<robot name="chess_{escape(kind)}">
  <link name="{escape(kind)}">
    <inertial>
      <origin xyz="{com[0]:.9g} {com[1]:.9g} {com[2]:.9g}" rpy="0 0 0"/>
      <mass value="{self.mass(kind):.9g}"/>
      <inertia ixx="{inertia[0, 0]:.9g}" ixy="{inertia[0, 1]:.9g}" ixz="{inertia[0, 2]:.9g}"
               iyy="{inertia[1, 1]:.9g}" iyz="{inertia[1, 2]:.9g}" izz="{inertia[2, 2]:.9g}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{escape(f"{kind}_visual.obj")}"/></geometry>
    </visual>
{collisions}
  </link>
</robot>
"""


##
# Board
##


class BoardSubset(NamedTuple):
    """One ``UsdGeom.Subset`` of the board -- in practice, the light or the dark squares."""

    name: str
    path: Path
    color: Vec3


class BoardMesh(NamedTuple):
    """A board's transcoded render meshes and its asset-local axis-aligned bounds.

    ``lower``/``upper`` already include ``board_scale`` on x and y, so they are the
    half-extents the collider box needs.
    """

    subsets: tuple[BoardSubset, ...]
    lower: np.ndarray  # (3,)
    upper: np.ndarray  # (3,)


@lru_cache(maxsize=None)
def board_geometry(board_usd: str, board_scale: float) -> BoardMesh:
    """Transcode a board asset into one OBJ per square colour, plus its bounds.

    The board scale is **baked into the exported vertices** rather than passed to Genesis
    as a morph scale, because it is anisotropic -- ``(s, s, 1.0)``, the same stretch the
    lab applies, since ``board_scale`` widens the squares and the board has no thickness to
    stretch.  ``gs.morphs.Mesh.scale`` is a single float, so the alternative would be to
    stretch the (zero) thickness too.  The cache is keyed on the scale, and a run uses one
    or two distinct values.

    ``board_usd`` may be a bare file name (resolved against ``assets/chess/board``) or a
    full path, so ``layout.board_usd`` and ``layout.board_usd_path`` both work.
    """
    path = Path(board_usd)
    if not path.is_absolute():
        path = BOARD_ASSET_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Missing board asset {path}. Fetch the LFS assets with: git lfs pull")

    key = {"board": path.name, "scale": round(float(board_scale), 6), "sources": _stamp(path)}
    directory, fresh = _cached_dir(f"boards/{path.stem}_s{board_scale:.3f}", key)

    stage = Usd.Stage.Open(str(path))
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    parts: list[tuple[str, np.ndarray, Vec3]] = []
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
            parts.append((child.GetName(), tris[np.isin(sources, faces)], _diffuse_color(child, BOARD_FALLBACK_COLOR)))
        triangles.append(tris)
        vertices.append(points)
        offset += len(points)

    points = np.concatenate(vertices)
    all_triangles = np.concatenate(triangles)
    if not parts:
        parts = [("squares", all_triangles, BOARD_FALLBACK_COLOR)]

    # (s, s, 1.0): board_scale stretches the squares, not the (zero) board thickness.
    scaled = points * np.array([board_scale, board_scale, 1.0])
    subsets = []
    for name, faces, color in parts:
        target = directory / f"{name}.obj"
        if not fresh or not target.exists():
            trimesh.Trimesh(vertices=scaled, faces=faces, process=False).export(target)
        subsets.append(BoardSubset(name=name, path=target, color=color))
    if not fresh:
        _write_manifest(directory, key)

    return BoardMesh(subsets=tuple(subsets), lower=scaled.min(0), upper=scaled.max(0))


class BoardPlacement(NamedTuple):
    """Where the board's render meshes and its collider box go, in world coordinates."""

    render_position: Vec3
    render_quat_xyzw: tuple[float, float, float, float]
    subsets: tuple[BoardSubset, ...]

    collider_position: Vec3
    collider_quat_xyzw: tuple[float, float, float, float]
    collider_size: Vec3
    """Full extents of the thin static box, not half-extents -- ``gs.morphs.Box`` takes
    ``size``."""


def board_placement(
    layout: BoardLayout,
    board_center: tuple[float, float] = BOARD_CENTER,
    table_top_z: float = TABLE_TOP_Z,
) -> BoardPlacement:
    """Resolve a layout into the meshes and the box that stand in for its board.

    The shipped board assets are zero-thickness quads.  Measured on the Newton port, which
    hit this first: as a mesh collider the pieces fall through (dz -1182 mm), as a convex
    hull they half fall through (-492 mm), as a thin box they rest (-1.7 mm).  So the
    physical board is a box whose *top* face sits exactly at ``table_top_z``, and the quads
    are render-only, lifted by :data:`~robochess_genesis.board_layout.BOARD_RENDER_LIFT` so
    they do not z-fight with the table top.
    """
    board = board_geometry(layout.board_usd, layout.board_scale)
    position, quat = board_asset_pose(layout, board_center, table_top_z)

    # The box is centred on the asset's own bounding box, expressed in the asset frame,
    # then carried into the world by the asset pose -- which for the 1D board is a 90
    # degree yaw, so the offset cannot simply be added componentwise.
    center_local = np.array(
        [
            (board.lower[0] + board.upper[0]) / 2.0,
            (board.lower[1] + board.upper[1]) / 2.0,
            -BOARD_THICKNESS / 2.0,
        ]
    )
    from .gsmath import quat_rotate  # local: keeps this module importable without numpy tricks

    center_world = np.asarray(position) + quat_rotate(np.asarray(quat), center_local)
    return BoardPlacement(
        render_position=(position[0], position[1], position[2] + BOARD_RENDER_LIFT),
        render_quat_xyzw=quat,
        subsets=board.subsets,
        collider_position=tuple(float(v) for v in center_world),
        collider_quat_xyzw=quat,
        collider_size=(
            float(board.upper[0] - board.lower[0]),
            float(board.upper[1] - board.lower[1]),
            BOARD_THICKNESS,
        ),
    )


def table_placement(
    center: tuple[float, float] = TABLE_CENTER,
    size: Vec3 = TABLE_SIZE,
    table_top_z: float = TABLE_TOP_Z,
) -> tuple[Vec3, Vec3]:
    """``(position, size)`` of the lab's static 1.30 x 1.10 x 0.77 m table.

    Its top face sits at ``table_top_z``; the box hangs down to the floor.
    """
    return (center[0], center[1], table_top_z - size[2] / 2.0), tuple(float(v) for v in size)


def piece_color(color: str | Sequence[float]) -> tuple[float, float, float, float]:
    """RGBA for a piece colour name (``"white"``/``"black"``) or a literal RGB triple."""
    rgb = PIECE_COLORS[color] if isinstance(color, str) else color
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def collision_pair_budget(n_pieces: int) -> int:
    """Broad-phase pair capacity **per environment**, for a board with this many pieces.

    Genesis' ``RigidOptions.max_collision_pairs`` defaults to 150, which is sized for a
    handful of primitives and is nowhere near a chess board: every piece carries 16 convex
    hulls, so a 32-piece 8x8 board is 512 piece geoms alone, and each one can pair with the
    board, the table and its neighbours' hulls.  Undersizing a broad-phase budget is the
    kind of failure that shows up as physics that is quietly *wrong* rather than as an
    error, which is exactly what bit the Newton port's contact budget, so this is sized
    generously and reported by ``ChessScene.describe()``.

    Per environment, not per scene: Genesis clamps the option to the env's own
    possible-pair count and shapes the contact cache ``(n_possible_pairs, n_envs)``, so
    multiplying by the batch size only inflates a number that is about to be clamped.

    ~90 pairs per piece is the measured peak with the 1.4x board stretch (neighbouring
    pieces are not touching, so most of the count is piece-vs-board); 256 per piece leaves
    the same margin the Newton port's 1600-contacts-per-piece did, with a floor that covers
    the arm's own geometry against the table on the small scenarios. Measured on 4x4: 210
    broad-phase pairs and 166 contacts actually used, against the 4096 budgeted.
    """
    return max(4096, 256 * int(n_pieces))
