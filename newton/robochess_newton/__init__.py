"""RoboChess on newton-physics: chess scenes, arms and GraspGen-driven picking."""

from __future__ import annotations

# board_layout owns the package's two import-time guards (PXR_WORK_THREAD_LIMIT and
# dropping the repo root from sys.path). Importing it here means a bare
# ``import robochess_newton`` installs them too, not just the submodules that use it.
from . import board_layout as board_layout  # noqa: F401
