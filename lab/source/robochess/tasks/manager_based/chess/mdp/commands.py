"""Command term that picks the chess move an episode has to execute."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import ChessMoveCommandCfg


class ChessMoveCommand(CommandTerm):
    """Samples "move piece *i* onto square *s*" at the start of every episode.

    Keeping the move in a command term (rather than in the demo script) means the
    task definition travels with the environment: a recorded episode, a replay and
    a future policy all see the same goal through the observation group, and
    :func:`~robochess.tasks.manager_based.chess.mdp.terminations.piece_on_target_square`
    can score it without any script-owned state.

    The command tensor is ``[target_x, target_y, target_z, piece_index]``, with the
    target expressed in the robot base frame.
    """

    cfg: ChessMoveCommandCfg

    def __init__(self, cfg: ChessMoveCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._pieces: list[RigidObject] = [env.scene[name] for name in cfg.piece_names]
        self._robot = env.scene[cfg.asset_name]

        self._square_pos = torch.tensor(cfg.target_positions, dtype=torch.float32, device=self.device)

        self._movable = torch.tensor(cfg.movable_piece_indices, dtype=torch.long, device=self.device)
        self._piece_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._square_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command = torch.zeros(self.num_envs, 4, device=self.device)

        self._preferred_kind: dict[int, str | None] = {}

        self.metrics["piece_to_target_distance"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"ChessMoveCommand: {len(self._movable)} movable pieces, "
            f"{len(self._square_pos)} destinations, resampled on reset."
        )

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Target square in the robot base frame plus the piece index. Shape is (num_envs, 4)."""
        return self._command

    @property
    def piece_index(self) -> torch.Tensor:
        """Index into ``cfg.piece_names`` of the piece to move. Shape is (num_envs,)."""
        return self._piece_index

    @property
    def target_pos_w(self) -> torch.Tensor:
        """Target square centre in the world frame. Shape is (num_envs, 3)."""
        return self._square_pos[self._square_index] + self._env.scene.env_origins

    @property
    def piece_pos_w(self) -> torch.Tensor:
        """Current position of the commanded piece in the world frame. Shape is (num_envs, 3)."""
        return self._stacked_piece_pos_w()[torch.arange(self.num_envs, device=self.device), self._piece_index]

    """
    Implementation.
    """

    def _stacked_piece_pos_w(self) -> torch.Tensor:
        return torch.stack([piece.data.root_pos_w.torch for piece in self._pieces], dim=1)

    def prefer_kinds(self, env_ids: torch.Tensor, kinds: Sequence[str]) -> None:
        """Restrict the next resample of ``env_ids`` to the given piece kinds.

        Lets a caller steer coverage -- "record a bishop next" -- without giving it
        the sampling logic. Uniform sampling needs a lot of episodes to touch all six
        kinds on a board that is half pawns; this makes a ten-demo run cover them.
        The preference lasts exactly one resample and then clears.
        """
        for env_id, kind in zip(env_ids.tolist(), list(kinds) + [None] * len(env_ids)):
            self._preferred_kind[env_id] = kind

    def _movable_of_kind(self, kind: str | None) -> torch.Tensor:
        if kind is None:
            return self._movable
        matching = [index for index in self._movable.tolist() if self.cfg.piece_kinds[index] == kind]
        return torch.tensor(matching, dtype=torch.long, device=self.device) if matching else self._movable

    def _resample_command(self, env_ids: Sequence[int]):
        indices = range(self.num_envs) if isinstance(env_ids, slice) else list(env_ids)
        for env_id in indices:
            movable = self._movable_of_kind(self._preferred_kind.get(int(env_id)))
            self._piece_index[env_id] = movable[torch.randint(len(movable), (1,), device=self.device)]
            self._preferred_kind.pop(int(env_id), None)
        self._square_index[env_ids] = torch.randint(len(self._square_pos), (len(indices),), device=self.device)

    def _update_command(self):
        self._command[:, :3] = math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w.torch, self._robot.data.root_quat_w.torch, self.target_pos_w
        )[0]
        self._command[:, 3] = self._piece_index.float()

    def _update_metrics(self):
        self.metrics["piece_to_target_distance"] = torch.norm(self.piece_pos_w - self.target_pos_w, dim=-1)
