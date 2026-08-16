"""Configuration for the chess move command."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils.configclass import configclass

from .commands import ChessMoveCommand


@configclass
class ChessMoveCommandCfg(CommandTermCfg):
    """Configuration for :class:`ChessMoveCommand`."""

    class_type: type = ChessMoveCommand

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """Effectively never resample mid-episode -- one move per episode."""

    asset_name: str = "robot"
    """Scene entity whose base frame the command is expressed in."""

    piece_names: list[str] = MISSING
    """Scene entity names of every piece on the board, in a fixed order."""

    piece_kinds: list[str] = MISSING
    """Piece kind (``pawn``, ``rook``, ...) for each entry of :attr:`piece_names`."""

    movable_piece_indices: list[int] = MISSING
    """Indices into :attr:`piece_names` that an episode may be asked to move."""

    target_positions: list[tuple[float, float, float]] = MISSING
    """Environment-frame resting positions a piece may be moved to.

    Free board squares plus any off-board slots (a capture tray). The full 4x4
    minichess setup has no empty square, so without the tray it would have nowhere
    legal to move a piece.
    """
