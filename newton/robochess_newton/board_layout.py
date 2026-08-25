"""Board, table and capture-tray geometry for the Newton port.

The scenario layouts themselves are *not* re-implemented here.  ``lab/source/robochess/
tasks/manager_based/chess/board.py`` is pure Python -- it imports nothing from Isaac Lab --
so this module loads that exact file and re-exports it.  Importing it the normal way
(``import robochess.tasks.manager_based.chess.board``) would drag in the package
``__init__`` chain, which pulls ``gymnasium`` and the Isaac Lab entry points; loading the
single file through :mod:`importlib` avoids that while keeping one source of truth for
square centres, piece placement and the 1D board's yaw/offset trick.

Everything else in this module is the *scene* geometry that lives in
``franka_chess_env_cfg.py`` on the Isaac Lab side: where the table top is, where the board
sits on it, and where captured pieces are parked.  Those numbers are duplicated (not
imported) because ``franka_chess_env_cfg.py`` imports ``isaaclab`` at module scope and
cannot be loaded outside an Isaac Sim app.

Frames follow the lab convention: ``+x`` runs along the ranks away from the robot, ``+y``
along the files to the robot's left, ``z`` up, and quaternions are ``(x, y, z, w)`` -- the
same order Isaac Lab 3.0 and ``warp`` both use, so poses transfer verbatim.

This module also holds the package's two import-time guards -- ``PXR_WORK_THREAD_LIMIT``
below and :func:`shield_newton_imports` -- because it is what the modules here import
before they touch ``pxr``, ``newton`` or anything that can reach them, and it itself
imports nothing beyond the standard library.  Both guards are idempotent, so a caller that
set them first loses nothing; what matters is that no import path can skip them, which
means any module here that does *not* import this one has to carry both guards itself.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType

# openusd 25.11's UsdPhysics parser races and segfaults nondeterministically when it is
# allowed a thread pool, which is what Usd.Stage.Open and ModelBuilder.add_usd() do under
# the hood. Measured: a bare pxr loop over one piece USD died after 5-23 iterations, three
# runs out of three. It has to be set before the process' first pxr import, hence module
# top of the package's mandatory-first import.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
LAB_BOARD_PY = REPO_ROOT / "lab" / "source" / "robochess" / "tasks" / "manager_based" / "chess" / "board.py"


def shield_newton_imports() -> None:
    """Drop the repo root from ``sys.path`` so ``newton/`` cannot shadow the real package.

    This repo has a top-level ``newton/`` directory -- the one holding this package -- and
    interpreters pick the repo root up routinely: ``python -c`` from the repo root, pytest
    rootdir insertion, ``PYTHONPATH=.``.  Today that directory has no ``__init__.py``, so
    PEP 420 makes it a mere namespace *portion* and the installed ``newton`` still wins
    (measured on both newton 1.2.1 and 1.6.0.dev0).  Add an ``__init__.py`` to ``newton/``
    and it becomes a regular package that beats site-packages outright: ``import newton``
    then succeeds and every ``newton.ModelBuilder`` lookup fails with a bewildering
    ``AttributeError``.  Removing the entry costs nothing and closes that door.

    Only entries that *resolve* to the repo root are dropped.  ``""`` and ``"."`` mean "the
    current directory", so they are resolved against it and survive unless the process is
    actually running from the repo root; removing ``""`` unconditionally would be a global,
    irreversible side effect on the host interpreter that breaks later cwd-relative imports
    in anything embedding this package, and it does not guard against anything extra.

    Called at import time by this module, which every other module in the package imports
    before it touches ``newton``.  Idempotent, so a script that ran the same filter itself
    before importing anything loses nothing by it running again.
    """
    repo = str(REPO_ROOT)
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != repo]

    module = sys.modules.get("newton")
    if module is None:
        return
    origin = getattr(module, "__file__", None) or next(iter(getattr(module, "__path__", [])), "")
    if origin and Path(origin).resolve().is_relative_to(REPO_ROOT):
        raise ImportError(
            f"'newton' was already imported from {origin}, which is this repo's own newton/ "
            "directory, not the newton physics package. Remove the repo root from sys.path "
            "before importing newton."
        )


shield_newton_imports()


def _load_lab_board() -> ModuleType:
    """Execute ``lab/.../chess/board.py`` as a standalone module.

    The module must be registered in ``sys.modules`` *before* ``exec_module``: its
    ``@dataclass`` decorators look their own module up by ``__name__`` to resolve string
    annotations, and raise ``AttributeError`` if it is not there yet.
    """
    name = f"{__name__}._lab_board"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LAB_BOARD_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_board = _load_lab_board()

BoardLayout = _board.BoardLayout
PieceSpec = _board.PieceSpec
make_layout = _board.make_layout
PIECE_KINDS: tuple[str, ...] = _board.PIECE_KINDS
SUPPORTED_SCENARIOS: tuple[str, ...] = _board.SUPPORTED_SCENARIOS
SQUARE_SIZE: float = _board.SQUARE_SIZE
BOARD_ASSET_DIR: Path = _board.BOARD_ASSET_DIR

##
# Table / board placement. Carried over from franka_chess_env_cfg.py, which cannot be
# imported outside an Isaac Sim app.
##

TABLE_TOP_Z = 0.77
"""Height [m] of the table top. Reference plane for the board, the pieces and the arm base."""

TABLE_SIZE = (1.30, 1.10, TABLE_TOP_Z)
TABLE_CENTER = (0.15, 0.0)
TABLE_EDGE_MARGIN = 0.06
"""Keep destinations this far inside the table edge, so a placed piece cannot topple off."""

BOARD_CENTER = (0.22, 0.0)
"""Legacy default board centre. The live value is ``board_distance`` in front of whichever
arm is selected -- the supported arms differ by more than 2x in reach."""

DEFAULT_BOARD_SCALE = {"pieces": 1.4, "1d": 1.4, "3x3": 1.4, "4x4": 1.4, "8x8": 1.0}
"""How much to stretch the squares, per scenario.

At scale 1.0 the 1.5x pieces fill 78-98% of a 60 mm square, leaving nowhere for a 22 mm
finger to descend beside a piece. 1.4 opens that to ~25 mm. The 8x8 board stays at 1.0
because stretching a 0.48 m board by 1.4 pushes most of it outside the Franka's reach."""

BOARD_THICKNESS = 0.010
"""Thickness [m] of the static box that stands in for the board.

The shipped board assets are zero-thickness quads at z=0. Used as a collider they let
every piece fall straight through (measured: dz = -1183 mm after 120 frames), so the
physical board is a thin box whose *top* face sits exactly at the table top."""

BOARD_RENDER_LIFT = 0.001
"""Lift [m] of the board render mesh above the table top, matching the lab's chessboard
prim. Without it the board quads are coplanar with the table top and z-fight."""

PIECE_SPAWN_CLEARANCE = 0.002
"""Pieces spawn this far above the board so they settle into contact instead of starting
interpenetrated. They drop ~1.7 mm: this clearance plus the 0.4-0.8 mm by which the CoACD
hulls undershoot z=0."""

PIECE_SCALE_TAG = "s150"
"""Which bake of ``lab/scripts/prepare_chess_assets.py`` to load (1.5x scale)."""

GENERATED_ASSET_DIR = REPO_ROOT / "assets" / "chess" / "generated"

PIECE_COLORS = {
    "white": (0.90, 0.88, 0.82),
    "black": (0.06, 0.06, 0.08),
}

TABLE_COLOR = (0.24, 0.18, 0.12)
CAPTURE_TRAY_COLOR = (0.45, 0.42, 0.38)

CAPTURE_TRAY_SHAPE = (2, 3)
"""Capture tray slot grid as ``(columns along x, rows along y)``."""

CAPTURE_TRAY_PITCH = 0.07
CAPTURE_TRAY_GAP = 0.09
"""Where captured pieces go, parked beside the board's -y edge with this much clearance.

The full 4x4 setup leaves no empty square, so the tray is the only legal destination
there; on the bigger boards it doubles the variety of place motions."""

CAPTURE_TRAY_THICKNESS = 0.001

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def yaw_quat(yaw: float) -> Quat:
    """Rotation about ``+z`` as an ``(x, y, z, w)`` quaternion."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def board_scale_for(scenario: str, board_scale: float | None = None) -> float:
    """Explicit scale if given, otherwise the per-scenario default."""
    if board_scale is not None:
        return board_scale
    if scenario not in DEFAULT_BOARD_SCALE:
        raise ValueError(f"Unsupported chess scenario: {scenario}. Choose one of {SUPPORTED_SCENARIOS}.")
    return DEFAULT_BOARD_SCALE[scenario]


def board_asset_pose(
    layout: BoardLayout,
    board_center: tuple[float, float] = BOARD_CENTER,
    table_top_z: float = TABLE_TOP_Z,
) -> tuple[Vec3, Quat]:
    """Pose of the board *asset's own origin* on the table top.

    Not the same as the board centre: the 1D board is modelled starting at its first cell
    and is turned 90 degrees to run left-to-right, so ``layout.board_prim_offset`` shifts
    it back by half its length. Verified against the assets' own quad centres -- every
    square lands within 3e-5 mm of ``layout.square_center``.

    The returned ``z`` is the playing surface. Render geometry is lifted by
    :data:`BOARD_RENDER_LIFT` on top of it; the collider box hangs below it.
    """
    offset_x, offset_y = layout.board_prim_offset
    position = (board_center[0] + offset_x, board_center[1] + offset_y, table_top_z)
    return position, yaw_quat(layout.board_prim_yaw)


def board_square_pos(
    layout: BoardLayout,
    file: int,
    rank: int,
    board_center: tuple[float, float] = BOARD_CENTER,
    table_top_z: float = TABLE_TOP_Z,
) -> Vec3:
    """Resting position of a piece standing on square ``(file, rank)``.

    ``square_center`` is in the board frame, which is axis-aligned with the world: the
    board yaw rotates the *asset*, never the layout grid.
    """
    x, y = layout.square_center(file, rank)
    return (board_center[0] + x, board_center[1] + y, table_top_z)


def piece_world_pose(
    layout: BoardLayout,
    spec: PieceSpec,
    board_center: tuple[float, float] = BOARD_CENTER,
    table_top_z: float = TABLE_TOP_Z,
) -> tuple[Vec3, Quat]:
    """Spawn pose of one piece: its starting square, lifted by the spawn clearance."""
    x, y, _ = board_square_pos(layout, spec.file, spec.rank, board_center, table_top_z)
    return (x, y, table_top_z + PIECE_SPAWN_CLEARANCE), yaw_quat(spec.yaw)


def capture_tray_center(
    layout: BoardLayout, board_center: tuple[float, float] = BOARD_CENTER
) -> tuple[float, float]:
    """Tray centre, parked clear of the board's -y edge."""
    tray_half = (CAPTURE_TRAY_SHAPE[1] * CAPTURE_TRAY_PITCH) / 2.0
    return board_center[0], board_center[1] - (layout.half_width + CAPTURE_TRAY_GAP + tray_half)


def capture_tray_slots(
    layout: BoardLayout, board_center: tuple[float, float] = BOARD_CENTER
) -> list[tuple[float, float]]:
    """World XY of every tray slot, column-major, matching the lab's slot ordering."""
    columns, rows = CAPTURE_TRAY_SHAPE
    center_x, center_y = capture_tray_center(layout, board_center)
    return [
        (
            center_x + (col - (columns - 1) / 2.0) * CAPTURE_TRAY_PITCH,
            center_y + (row - (rows - 1) / 2.0) * CAPTURE_TRAY_PITCH,
        )
        for col in range(columns)
        for row in range(rows)
    ]


def capture_tray_size() -> Vec3:
    """Full extents of the tray's render slab."""
    return (
        CAPTURE_TRAY_SHAPE[0] * CAPTURE_TRAY_PITCH,
        CAPTURE_TRAY_SHAPE[1] * CAPTURE_TRAY_PITCH,
        CAPTURE_TRAY_THICKNESS,
    )


def is_on_table(x: float, y: float) -> bool:
    """Whether a destination is far enough inside the table edge to be safe to release on.

    On a wide board the tray gets pushed out far enough that some slots hang over the
    edge, where a released piece just falls off.
    """
    return (
        abs(x - TABLE_CENTER[0]) <= TABLE_SIZE[0] / 2.0 - TABLE_EDGE_MARGIN
        and abs(y - TABLE_CENTER[1]) <= TABLE_SIZE[1] / 2.0 - TABLE_EDGE_MARGIN
    )
