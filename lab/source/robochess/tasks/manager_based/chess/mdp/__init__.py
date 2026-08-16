"""MDP terms for the RoboChess manipulation tasks.

Re-exports the stock Isaac Lab terms so an env config can pull everything from one
module, then adds the chess-specific commands, observations and terminations.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .commands import ChessMoveCommand  # noqa: F401
from .commands_cfg import ChessMoveCommandCfg  # noqa: F401
from .events import reset_pieces_on_board  # noqa: F401
from .observations import (  # noqa: F401
    commanded_piece_pose,
    ee_frame_pos,
    ee_frame_quat,
    gripper_open_fraction,
    gripper_pos,
    piece_grasped,
    piece_lifted,
    piece_orientations,
    piece_placed,
    piece_positions,
)
from .terminations import any_piece_off_board, any_piece_toppled, move_completed  # noqa: F401
