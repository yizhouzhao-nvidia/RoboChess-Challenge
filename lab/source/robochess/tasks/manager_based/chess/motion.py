"""Motion primitives shared by the single-arm demo generator and the game driver.

Extracted so the two scripts cannot drift apart: the phase schedule, the carry-height
rule and the GraspGen retargeting are exactly the parts where a silent difference
between "recording a pick" and "playing a game" would produce two subtly different
datasets.

Everything here is arm-agnostic -- it takes a measured
:class:`~robochess.tasks.manager_based.chess.robot_configs.ChessRobotSpec` and works
out the rest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import isaaclab.utils.math as math_utils

from .robot_configs import ChessRobotSpec

# Gripper geometry. Modelled from the arm's measured ChessRobotSpec rather than
# hardcoded, so the neighbour-clearance check works for any of the four arms.
##

FINGER_THICKNESS = 0.013
FINGER_LENGTH = 0.055

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0



class Phase:
    """One leg of the pick-and-place: interpolate to ``goal``, then wait for the arm to arrive.

    Differential IK takes one Jacobian step per control tick, so a purely
    time-triggered schedule closes the fingers wherever the arm happens to be --
    tens of millimetres short of the grasp, which slides the piece out. Each phase
    therefore also holds at its goal until the pose error is inside tolerance
    (or ``settle_timeout`` expires, so a bad IK solution cannot stall the episode).
    """

    def __init__(
        self,
        name: str,
        duration: float,
        gripper: float,
        goal: str,
        pos_tolerance: float = 0.004,
        rot_tolerance: float = 0.06,
        settle_timeout: float = 1.5,
    ):
        self.name = name
        self.duration = duration
        self.gripper = gripper
        self.goal = goal
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance
        self.settle_timeout = settle_timeout


PHASES = (
    Phase("pre_grasp", 1.8, GRIPPER_OPEN, "pre_grasp", pos_tolerance=0.006),
    Phase("descend", 1.0, GRIPPER_OPEN, "grasp", pos_tolerance=0.003, rot_tolerance=0.04, settle_timeout=2.0),
    Phase("close", 0.9, GRIPPER_CLOSE, "grasp", settle_timeout=0.0),
    Phase("lift", 1.0, GRIPPER_CLOSE, "lift", pos_tolerance=0.008),
    Phase("transfer", 2.2, GRIPPER_CLOSE, "pre_place", pos_tolerance=0.008),
    Phase("place", 1.2, GRIPPER_CLOSE, "place", pos_tolerance=0.003, rot_tolerance=0.04, settle_timeout=2.0),
    Phase("release", 0.6, GRIPPER_OPEN, "place", settle_timeout=0.0),
    Phase("retreat", 1.0, GRIPPER_OPEN, "pre_place", pos_tolerance=0.02),
    Phase("settle", 0.8, GRIPPER_OPEN, "pre_place", settle_timeout=0.0),
)

APPROACH_STANDOFF = 0.09
"""How far back along the approach axis the pre-grasp pose sits [m]."""

CARRY_CLEARANCE = 0.05
"""Gap left between the underside of the carried piece and the tallest piece on the board [m].

The carried piece hangs below the hand, so the carry height has to be set by what it
has to fly over -- a fixed 0.14 m lift leaves a pawn's base level with the top of the
king and drags it off the board.
"""

MIN_LIFT_HEIGHT = 0.12
"""Floor on the carry height [m], for boards whose pieces are all short."""

CARRY_REACH_FRACTION = 0.36
"""Cap on the carry height as a fraction of the arm's reach."""

PLACE_APPROACH_HEIGHT = 0.12
"""How high above the destination the carry ends, before the final descent [m].

Deliberately lower than the carry height. The destination is always an empty square
or an empty tray slot, so nothing has to be cleared there -- and the far tray slots
sit at the edge of the Franka's reach, where asking for the full carry height leaves
the arm 80-120 mm short of its target and the piece lands off-square.
"""

PLACE_CLEARANCE = 0.004
"""Gap left under the piece when releasing, so it drops rather than being ground in [m]."""


def least_attempted_kinds(attempts: dict[str, int], available: list[str]) -> list[str]:
    """Piece kinds to try next, least-attempted first -- i.e. round-robin.

    Deliberately steers on *attempts*, not successes. Attributing a success to the
    environment that earned it turns out to be unreliable after the fact: the env
    auto-resets inside ``step()``, so by the time the caller sees the done flags the
    command has been resampled and the termination manager recomputed, and the
    recorder's exported count cannot be split across a batch that ended together.
    Attempts are counted from a plan-time snapshot, so they are exact.

    Round-robin also cannot fixate. Ranking by successes does: a kind the arm simply
    cannot pick stays the scarcest forever and absorbs every episode -- one run spent
    182 of 260 attempts on a single piece and returned a 2% success rate.
    """
    present = [kind for kind in attempts if kind in set(available)]
    return sorted(present, key=lambda kind: attempts[kind])


def slerp(quat_a: torch.Tensor, quat_b: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Batched geodesic interpolation between two (x, y, z, w) quaternions."""
    return math_utils.quat_box_plus(quat_a, tau.unsqueeze(-1) * math_utils.quat_box_minus(quat_b, quat_a))


def rotation_z(angle: torch.Tensor) -> torch.Tensor:
    """Batched 4x4 rotation about Z."""
    matrices = torch.eye(4, device=angle.device).repeat(*angle.shape, 1, 1)
    cos, sin = torch.cos(angle), torch.sin(angle)
    matrices[..., 0, 0] = cos
    matrices[..., 0, 1] = -sin
    matrices[..., 1, 0] = sin
    matrices[..., 1, 1] = cos
    return matrices


def pose_to_matrix(position: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """(N, 3) + (N, 4) xyzw -> (N, 4, 4)."""
    matrices = torch.eye(4, device=position.device).repeat(position.shape[0], 1, 1)
    matrices[:, :3, :3] = math_utils.matrix_from_quat(quaternion)
    matrices[:, :3, 3] = position
    return matrices


def matrix_to_pose(matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 4, 4) -> (N, 3) + (N, 4) xyzw."""
    return matrices[:, :3, 3], math_utils.quat_from_matrix(matrices[:, :3, :3])


def gripper_probe_points(spec: ChessRobotSpec, device: torch.device) -> torch.Tensor:
    """Sample points on the two open fingers, in the end-effector frame. Shape is (P, 3).

    A slab per finger, running back from the TCP along the approach axis and offset
    to either side along the closing axis. Crude, but it is only used to ask "would
    these fingers dip into a neighbouring piece", and it needs no per-arm mesh.
    """
    approach = torch.tensor(spec.approach_axis, device=device, dtype=torch.float32)
    closing = torch.tensor(spec.closing_axis, device=device, dtype=torch.float32)
    lateral = torch.cross(approach, closing, dim=0)
    tcp = torch.tensor(spec.tcp_offset, device=device, dtype=torch.float32)

    half = spec.max_opening / 2.0
    points = []
    for side in (1.0, -1.0):
        for offset in (half, half + FINGER_THICKNESS):
            for back in (0.0, 0.5 * FINGER_LENGTH, FINGER_LENGTH):
                for across in (-FINGER_THICKNESS, FINGER_THICKNESS):
                    points.append(tcp - approach * back + closing * (side * offset) + lateral * across)
    return torch.stack(points)


class GraspLibrary:
    """GraspGen grasps per piece kind, retargeted to one arm's end-effector frame.

    GraspGen only ships models for three grippers, but a chess grasp is a top-down
    pinch of a shaft -- a property of the piece and of parallel-jaw geometry. So the
    franka_panda grasps are reused for every arm and simply re-expressed in its
    end-effector frame, which is what ``ChessRobotSpec.graspgen_to_ee`` encodes.
    """

    def __init__(self, grasp_file: Path, spec: ChessRobotSpec, device: torch.device, max_candidates: int):
        payload = json.loads(grasp_file.read_text())
        if payload.get("gripper") != "franka_panda":
            raise ValueError(f"{grasp_file} was generated for gripper '{payload.get('gripper')}', expected franka_panda")

        convention_fix = torch.tensor(spec.graspgen_to_ee(payload["gripper_depth"]), dtype=torch.float32, device=device)
        self.piece_scale = payload["piece_scale"]
        self.hand_in_piece: dict[str, torch.Tensor] = {}
        self.scores: dict[str, torch.Tensor] = {}
        self.yaw_free: dict[str, bool] = {}
        for kind, result in payload["pieces"].items():
            records = result["grasps"][:max_candidates]
            matrices = torch.tensor(
                [record["matrix"] for record in records], dtype=torch.float32, device=device
            ).reshape(-1, 4, 4)
            self.hand_in_piece[kind] = matrices @ convention_fix
            self.scores[kind] = torch.tensor([record["rank_score"] for record in records], device=device)
            self.yaw_free[kind] = bool(result["yaw_free"])
        print(f"[INFO] Loaded GraspGen grasps for {sorted(self.hand_in_piece)} (piece scale {self.piece_scale}x)")

    def candidates(self, kind: str, num_yaws: int) -> tuple[torch.Tensor, torch.Tensor]:
        """All (grasp x yaw) hand poses in the piece frame, plus their scores."""
        matrices = self.hand_in_piece[kind]
        scores = self.scores[kind]
        if not self.yaw_free[kind]:
            return matrices, scores
        yaws = torch.arange(num_yaws, device=matrices.device) * (2.0 * math.pi / num_yaws)
        spun = rotation_z(yaws).unsqueeze(1) @ matrices.unsqueeze(0)
        return spun.reshape(-1, 4, 4), scores.repeat(num_yaws)
