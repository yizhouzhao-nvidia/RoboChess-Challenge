"""Legal-move rules for the three RoboChess game variants.

Deliberately free of Isaac Lab, torch and USD: the rules are the part of a
robot-vs-robot game most likely to be wrong, and keeping them pure Python means
they can be exhaustively tested in milliseconds without a simulator
(``lab/scripts/test_chess_rules.py``).

Piece identity is an *index into the layout's piece tuple*, which is the same
index the scene uses for ``piece_<color>_<kind>_<n>``. A move therefore names
exactly which rigid body the arm has to pick up, with no lookup in between.

Three variants, all played without castling or en passant:

``1d``
    One rank of eight cells. White ``K N R`` on cells 0-2 faces black ``R N K``
    on cells 5-7. The king steps one cell, the rook slides, and the knight jumps
    exactly two cells *over* whatever is in between. Won by checkmate.
``hexapawn``
    Three pawns each on a 3x3 board. Pawns step one square forward or capture one
    square diagonally forward. Won by reaching the far rank, capturing every enemy
    pawn, or leaving the opponent with no legal move.
``minichess``
    4x4 with a mirrored back rank (``K N N R`` against ``R N N K``) and a rank of
    pawns each. Ordinary chess movement clipped to the small board; pawns take a
    single step and promote to a queen on the far rank. Won by checkmate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

Square = tuple[int, int]
"""``(file, rank)``, matching :meth:`BoardLayout.square_center`."""

VARIANTS = ("1d", "hexapawn", "minichess")

SLIDERS = {"rook": ((1, 0), (-1, 0), (0, 1), (0, -1)),
           "bishop": ((1, 1), (1, -1), (-1, 1), (-1, -1))}
SLIDERS["queen"] = SLIDERS["rook"] + SLIDERS["bishop"]

KING_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_STEPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))

FORWARD = {"white": 1, "black": -1}
"""Direction a pawn advances in, along the rank axis."""


def opponent(color: str) -> str:
    return "black" if color == "white" else "white"


@dataclass(frozen=True)
class Move:
    """One ply: move ``piece`` from ``origin`` to ``target``.

    ``captured`` is the index of the piece removed by this move, which the robot
    has to lift off the board *before* the moving piece can be placed. ``promotion``
    names the kind the piece becomes on arrival, which in simulation means swapping
    one rigid body for another rather than moving one.
    """

    piece: int
    origin: Square
    target: Square
    captured: int | None = None
    promotion: str | None = None

    def __str__(self) -> str:
        arrow = "x" if self.captured is not None else "-"
        text = f"{self.origin[0]},{self.origin[1]}{arrow}{self.target[0]},{self.target[1]}"
        return f"{text}={self.promotion}" if self.promotion else text


@dataclass(frozen=True)
class Piece:
    color: str
    kind: str
    square: Square | None
    """``None`` once captured; the index stays valid so the scene mapping holds."""

    @property
    def alive(self) -> bool:
        return self.square is not None


class Position:
    """A game position: piece list, side to move, and the rules of one variant.

    Positions are treated as immutable -- :meth:`apply` returns a new one -- so a
    caller can search or roll back without copying by hand.
    """

    def __init__(self, variant: str, pieces: list[Piece], side_to_move: str = "white",
                 num_files: int = 0, num_ranks: int = 0, history: tuple[str, ...] = ()):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose one of {VARIANTS}.")
        self.variant = variant
        self.pieces = pieces
        self.side_to_move = side_to_move
        self.num_files = num_files
        self.num_ranks = num_ranks
        self.history = history

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_layout(cls, layout, variant: str | None = None) -> "Position":
        """Build the starting position from a :class:`BoardLayout`.

        The piece order is preserved exactly, so ``Move.piece`` indexes both
        ``layout.pieces`` and the scene entity list.
        """
        variant = variant or scenario_variant(layout.scenario)
        pieces = [Piece(spec.color, spec.kind, (spec.file, spec.rank)) for spec in layout.pieces]
        return cls(variant, pieces, "white", layout.num_files, layout.num_ranks)

    # ------------------------------------------------------------------ board queries
    def on_board(self, square: Square) -> bool:
        return 0 <= square[0] < self.num_files and 0 <= square[1] < self.num_ranks

    def at(self, square: Square) -> int | None:
        """Index of the piece standing on ``square``, if any."""
        for index, piece in enumerate(self.pieces):
            if piece.square == square:
                return index
        return None

    def king_index(self, color: str) -> int | None:
        for index, piece in enumerate(self.pieces):
            if piece.alive and piece.color == color and piece.kind == "king":
                return index
        return None

    def key(self) -> str:
        """Position fingerprint, for repetition detection."""
        board = ",".join(
            f"{p.color[0]}{p.kind[0]}{p.square[0]}{p.square[1]}" if p.alive else "-"
            for p in self.pieces
        )
        return f"{board}|{self.side_to_move}"

    # ------------------------------------------------------------------ move generation
    def _pseudo_legal(self, color: str) -> list[Move]:
        """Moves that respect piece movement but may leave the king in check."""
        moves: list[Move] = []
        for index, piece in enumerate(self.pieces):
            if not piece.alive or piece.color != color:
                continue
            moves += self._piece_moves(index, piece)
        return moves

    def _piece_moves(self, index: int, piece: Piece) -> list[Move]:
        assert piece.square is not None
        if piece.kind == "pawn":
            return self._pawn_moves(index, piece)
        if piece.kind == "knight":
            return self._hop_moves(index, piece, self._knight_steps())
        if piece.kind == "king":
            return self._hop_moves(index, piece, self._king_steps())
        return self._slide_moves(index, piece, SLIDERS[piece.kind])

    def _knight_steps(self):
        # On a single rank an L-shape has nowhere to go, so 1D chess defines the
        # knight as a two-cell jump along the line instead.
        return ((2, 0), (-2, 0)) if self.variant == "1d" else KNIGHT_STEPS

    def _king_steps(self):
        return ((1, 0), (-1, 0)) if self.variant == "1d" else KING_STEPS

    def _hop_moves(self, index: int, piece: Piece, steps) -> list[Move]:
        """Jumping moves: blockers in between are irrelevant, only the landing square."""
        moves = []
        file, rank = piece.square
        for dx, dy in steps:
            target = (file + dx, rank + dy)
            if not self.on_board(target):
                continue
            occupant = self.at(target)
            if occupant is None:
                moves.append(Move(index, piece.square, target))
            elif self.pieces[occupant].color != piece.color:
                moves.append(Move(index, piece.square, target, captured=occupant))
        return moves

    def _slide_moves(self, index: int, piece: Piece, directions) -> list[Move]:
        moves = []
        file, rank = piece.square
        for dx, dy in directions:
            step = 1
            while True:
                target = (file + dx * step, rank + dy * step)
                if not self.on_board(target):
                    break
                occupant = self.at(target)
                if occupant is None:
                    moves.append(Move(index, piece.square, target))
                else:
                    if self.pieces[occupant].color != piece.color:
                        moves.append(Move(index, piece.square, target, captured=occupant))
                    break
                step += 1
        return moves

    def _pawn_moves(self, index: int, piece: Piece) -> list[Move]:
        moves = []
        file, rank = piece.square
        step = FORWARD[piece.color]
        last_rank = self.num_ranks - 1 if piece.color == "white" else 0

        ahead = (file, rank + step)
        if self.on_board(ahead) and self.at(ahead) is None:
            promotion = "queen" if self.variant == "minichess" and ahead[1] == last_rank else None
            moves.append(Move(index, piece.square, ahead, promotion=promotion))

        for dx in (-1, 1):
            diagonal = (file + dx, rank + step)
            if not self.on_board(diagonal):
                continue
            occupant = self.at(diagonal)
            if occupant is not None and self.pieces[occupant].color != piece.color:
                promotion = "queen" if self.variant == "minichess" and diagonal[1] == last_rank else None
                moves.append(Move(index, piece.square, diagonal, captured=occupant, promotion=promotion))
        return moves

    def legal_moves(self, color: str | None = None) -> list[Move]:
        """Pseudo-legal moves minus those that leave one's own king attacked.

        Hexapawn has no king, so every pseudo-legal move is legal.
        """
        color = color or self.side_to_move
        pseudo = self._pseudo_legal(color)
        if self.variant == "hexapawn":
            return pseudo
        return [move for move in pseudo if not self.apply(move).is_check(color)]

    # ------------------------------------------------------------------ game state
    def is_attacked(self, square: Square, by: str) -> bool:
        return any(move.target == square for move in self._pseudo_legal(by))

    def is_check(self, color: str) -> bool:
        king = self.king_index(color)
        if king is None:
            return False
        return self.is_attacked(self.pieces[king].square, opponent(color))

    def apply(self, move: Move) -> "Position":
        pieces = list(self.pieces)
        if move.captured is not None:
            pieces[move.captured] = replace(pieces[move.captured], square=None)
        moved = pieces[move.piece]
        pieces[move.piece] = replace(
            moved, square=move.target, kind=move.promotion or moved.kind
        )
        return Position(self.variant, pieces, opponent(self.side_to_move),
                        self.num_files, self.num_ranks, self.history + (self.key(),))

    def result(self) -> str | None:
        """``'white'``, ``'black'``, ``'draw'``, or ``None`` if the game continues."""
        if self.variant == "hexapawn":
            for color in ("white", "black"):
                far = self.num_ranks - 1 if color == "white" else 0
                if any(p.alive and p.color == color and p.square[1] == far for p in self.pieces):
                    return color
                if not any(p.alive and p.color == opponent(color) for p in self.pieces):
                    return color
            # A player with no move loses, which is the opposite of chess stalemate.
            if not self.legal_moves():
                return opponent(self.side_to_move)
            return None

        if not self.legal_moves():
            return opponent(self.side_to_move) if self.is_check(self.side_to_move) else "draw"
        if self.history.count(self.key()) >= 2:
            return "draw"
        if all(not p.alive or p.kind == "king" for p in self.pieces):
            return "draw"
        return None


def scenario_variant(scenario: str) -> str:
    """Map a :mod:`board` scenario name onto the variant that governs it."""
    mapping = {"1d": "1d", "3x3": "hexapawn", "minichess": "minichess"}
    if scenario not in mapping:
        raise ValueError(f"Scenario '{scenario}' is not playable as a game. Playable: {sorted(mapping)}")
    return mapping[scenario]
