"""Board geometry shared by the RoboChess manipulation tasks.

The shipped boards in ``assets/chess/board`` all use 60 mm squares at scale 1.0
(the 1D board is 8 cells long).  A layout maps the four scenarios onto a common
(file, rank) grid so a task can ask for "the centre of square (2, 0)" without
caring which board is loaded, and can stretch the squares via ``board_scale``
without touching the pieces.

Board frame convention, chosen so the arm faces the board head-on:

* ``+x`` runs along the ranks, away from the robot,
* ``+y`` runs along the files, to the robot's left,
* ``z = 0`` is the playing surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[6] / "assets"
CHESS_ASSET_DIR = ASSET_DIR / "chess"
BOARD_ASSET_DIR = CHESS_ASSET_DIR / "board"

SQUARE_SIZE = 0.06
"""Edge length [m] of one square on the shipped boards at scale 1.0."""

PIECE_KINDS = ("pawn", "rook", "knight", "bishop", "queen", "king")
"""Piece assets available under ``assets/chess`` (and baked into ``generated/``)."""

SUPPORTED_SCENARIOS = ("pieces", "1d", "3x3", "4x4", "8x8")


@dataclass(frozen=True)
class PieceSpec:
    """One piece placed on the board at the start of an episode."""

    color: str
    kind: str
    index: int
    file: int
    rank: int
    yaw: float = 0.0

    @property
    def name(self) -> str:
        """Scene entity name, e.g. ``piece_white_pawn_0``."""
        return f"piece_{self.color}_{self.kind}_{self.index}"


@dataclass(frozen=True)
class BoardLayout:
    """A scenario: which board asset to load and where the pieces start."""

    scenario: str
    board_usd: str
    num_files: int
    num_ranks: int
    pieces: tuple[PieceSpec, ...]
    square_size: float = SQUARE_SIZE
    board_scale: float = 1.0

    @property
    def board_usd_path(self) -> str:
        return str(BOARD_ASSET_DIR / self.board_usd)

    def square_center(self, file: int, rank: int) -> tuple[float, float]:
        """Centre of square ``(file, rank)`` in the board frame."""
        x = (rank - (self.num_ranks - 1) / 2.0) * self.square_size
        y = (file - (self.num_files - 1) / 2.0) * self.square_size
        return x, y

    @property
    def half_width(self) -> float:
        """Half the board's extent along y (the files)."""
        return self.num_files * self.square_size / 2.0

    @property
    def board_prim_yaw(self) -> float:
        """Rotation of the board asset about z.

        The 1D board is modelled as a row running along +x. Laid out that way in
        front of the arm, its two ends fall outside the Franka's comfortable reach,
        so it is turned to run left-to-right instead, at a constant distance.
        """
        return math.pi / 2 if self.scenario == "1d" else 0.0

    @property
    def board_prim_offset(self) -> tuple[float, float]:
        """Offset from the board frame origin to the board asset's own origin.

        The n x n boards are modelled around their centre, but the 1D board starts
        at its first cell, so it has to be shifted back by half its length -- along
        y, because :attr:`board_prim_yaw` has turned it.
        """
        if self.scenario == "1d":
            return 0.0, -(self.num_files - 1) / 2.0 * self.square_size
        return 0.0, 0.0

    def free_squares(self) -> list[tuple[int, int]]:
        """Squares with no piece on them at reset."""
        occupied = {(piece.file, piece.rank) for piece in self.pieces}
        return [
            (file, rank)
            for rank in range(self.num_ranks)
            for file in range(self.num_files)
            if (file, rank) not in occupied
        ]


_KNIGHT_YAW = {"white": math.pi / 2, "black": -math.pi / 2}
_BACK_RANK_4X4 = ("king", "knight", "knight", "rook")
_BACK_RANK_8X8 = ("rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook")


def _back_rank(color: str, kinds: tuple[str, ...], rank: int, counts: dict[str, int]) -> list[PieceSpec]:
    pieces = []
    for file, kind in enumerate(kinds):
        index = counts.get((color, kind), 0)
        counts[(color, kind)] = index + 1
        pieces.append(PieceSpec(color, kind, index, file, rank, _KNIGHT_YAW[color] if kind == "knight" else 0.0))
    return pieces


def make_layout(scenario: str, board_scale: float = 1.0) -> BoardLayout:
    """Build the :class:`BoardLayout` for one of :data:`SUPPORTED_SCENARIOS`.

    ``board_scale`` stretches the squares without touching the pieces. At scale 1.0
    the shipped pieces fill 78-98% of a square, which leaves almost nowhere for the
    gripper fingers to descend beside a piece; scaling the board up is the cheapest
    way to open that gap.
    """
    square = SQUARE_SIZE * board_scale
    common = {"square_size": square, "board_scale": board_scale}

    if scenario == "pieces":
        # One of every kind on a 3x3 board, back two ranks, front rank left empty as
        # destinations. Exists so a short run can demonstrate picking all six pieces
        # -- the standard setups are mostly pawns, and 8x8 is far too wide for the
        # smaller arms to reach.
        order = (("rook", "knight", "bishop"), ("queen", "king", "pawn"))
        pieces = tuple(
            PieceSpec("white", kind, 0, file, rank + 1, _KNIGHT_YAW["white"] if kind == "knight" else 0.0)
            for rank, row in enumerate(order)
            for file, kind in enumerate(row)
        )
        return BoardLayout("pieces", "board_3x3.usdc", num_files=3, num_ranks=3, pieces=pieces, **common)

    if scenario == "1d":
        # A single 8-cell rank: white king/knight/rook facing black rook/knight/king.
        placements = (
            ("white", "king", 0), ("white", "knight", 1), ("white", "rook", 2),
            ("black", "rook", 5), ("black", "knight", 6), ("black", "king", 7),
        )
        pieces = tuple(
            PieceSpec(color, kind, 0, file, 0, _KNIGHT_YAW[color] if kind == "knight" else 0.0)
            for color, kind, file in placements
        )
        return BoardLayout("1d", "board_1x6_large.usdc", num_files=8, num_ranks=1, pieces=pieces, **common)

    if scenario == "3x3":
        pieces = tuple(
            PieceSpec(color, "pawn", file, file, rank)
            for color, rank in (("white", 0), ("black", 2))
            for file in range(3)
        )
        return BoardLayout("3x3", "board_3x3.usdc", num_files=3, num_ranks=3, pieces=pieces, **common)

    if scenario == "4x4":
        counts: dict[tuple[str, str], int] = {}
        pieces: list[PieceSpec] = []
        for color, back, pawn in (("white", 0, 1), ("black", 3, 2)):
            pieces += _back_rank(color, _BACK_RANK_4X4, back, counts)
            pieces += [PieceSpec(color, "pawn", file, file, pawn) for file in range(4)]
        return BoardLayout("4x4", "board_4x4.usdc", num_files=4, num_ranks=4, pieces=tuple(pieces), **common)

    if scenario == "8x8":
        counts = {}
        pieces = []
        for color, back, pawn in (("white", 0, 1), ("black", 7, 6)):
            pieces += _back_rank(color, _BACK_RANK_8X8, back, counts)
            pieces += [PieceSpec(color, "pawn", file, file, pawn) for file in range(8)]
        return BoardLayout("8x8", "board_8x8.usdc", num_files=8, num_ranks=8, pieces=tuple(pieces), **common)

    raise ValueError(f"Unsupported chess scenario: {scenario}. Choose one of {SUPPORTED_SCENARIOS}.")
