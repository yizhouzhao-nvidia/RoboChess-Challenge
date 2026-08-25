"""GraspGen grasps for the chess pieces, retargeted to whichever arm is driving.

GraspGen ships discriminator models for three grippers only, so the grasps in
``assets/chess/generated/<scale>/grasps/chess_grasps.json`` were generated for the
Franka panda hand.  That is not the limitation it looks like: a chess grasp is a
top-down pinch of a shaft, which is a property of the *piece* and of parallel-jaw
geometry, not of a particular hand.  So the franka_panda grasps are reused for
every arm and simply re-expressed in that arm's end-effector frame -- which is
exactly what :meth:`NewtonRobotSpec.graspgen_to_ee` encodes.

Two things happen here, both ported from
``lab/scripts/generate_chess_pick_demos.py``:

* :class:`GraspLibrary` loads the JSON, retargets every candidate into the arm's
  ``ee_body`` frame and spins the yaw-free pieces (everything but the knight is a
  solid of revolution, so any rotation about the piece axis is an equally valid
  grasp and multiplies 12 candidates into 192);
* :class:`GraspPlanner` re-scores those candidates against the board as it
  actually stands at the start of the episode.  GraspGen scored each piece in
  isolation, and the best grasp of a lone bishop is frequently the one that rakes
  the open fingers through the queen next to it.  The neighbours are modelled as
  upright cylinders and the 24-point probe cloud of the open gripper is tested
  against them, at the grasp pose *and* at the pre-grasp pose 90 mm back along the
  approach, because the fingers have to get there too.

Everything in this module is plain numpy on the host: it runs once per episode
per world (about 300 us for 192 candidates against 31 neighbours), not per tick.
Quaternions are ``(x, y, z, w)`` throughout, matching warp and Isaac Lab 3.0, and
poses are 4x4 row-major homogeneous matrices, matching the JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

# board_layout before anything else: importing it strips the repo root from
# sys.path, where this repo's own newton/ directory would shadow the real package.
from . import board_layout as bl

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module newton-free
    from .robots import NewtonRobotSpec

__all__ = [
    "APPROACH_STANDOFF",
    "CARRY_CLEARANCE",
    "CARRY_REACH_FRACTION",
    "CLEARANCE_MARGIN",
    "COLLISION_WEIGHT",
    "FINGER_LENGTH",
    "FINGER_THICKNESS",
    "GOAL_NAMES",
    "MIN_LIFT_HEIGHT",
    "PLACE_APPROACH_HEIGHT",
    "PLACE_CLEARANCE",
    "GraspLibrary",
    "GraspPlan",
    "GraspPlanner",
    "carry_height",
    "default_grasp_file",
    "gripper_probe_points",
    "load_piece_geometry",
    "matrix_to_pose",
    "place_goals",
    "pose_to_matrix",
    "quat_error_magnitude",
    "quat_from_matrix",
    "quat_inverse",
    "quat_mul",
    "quat_rotate",
    "quat_slerp",
    "quat_to_matrix",
    "rotation_z",
    "yaw_quat_of",
]

##
# Gripper geometry and scoring weights, all from generate_chess_pick_demos.py.
##

FINGER_THICKNESS = 0.013
"""Half-width of the modelled finger slab [m]."""

FINGER_LENGTH = 0.055
"""How far the finger slab runs back from the TCP along ``-approach`` [m]."""

APPROACH_STANDOFF = 0.09
"""How far back along the approach axis the pre-grasp pose sits [m]."""

CLEARANCE_MARGIN = 0.004
"""Neighbour cylinders are inflated by this much before the overlap test [m].

A grasp that merely grazes a neighbour still knocks it over once the fingers
close, so the penalty has to start before contact does."""

COLLISION_WEIGHT = 40.0
"""Score charged per metre of worst-case penetration [1/m].

1 mm of overlap costs 0.04 of rank score and 25 mm costs 1.0, which is more than
the entire spread of rank scores -- i.e. a deep collision always loses to a
merely mediocre grasp, and a sub-millimetre graze breaks ties."""

CARRY_CLEARANCE = 0.05
"""Gap between the underside of the carried piece and the tallest piece still standing [m]."""

MIN_LIFT_HEIGHT = 0.12
"""Floor on the carry height [m], for boards whose pieces are all short."""

CARRY_REACH_FRACTION = 0.36
"""Cap on the carry height as a fraction of the arm's reach.

A carry sized for a Panda puts a shorter arm's wrist outside its workspace, where
the IK gives up and the piece is dropped mid-transfer."""

PLACE_APPROACH_HEIGHT = 0.12
"""Fallback height above the destination at which the carry ends [m].

The Isaac Lab script hard-codes this, and it is a trap on the wider boards: the
destination itself is always empty, but the *path* to it is not.  The carry is
sized by :func:`carry_height` to clear the tallest piece standing (191 mm with a
king on the board), while ``transfer`` interpolates straight from the lift to the
pre-place -- so a fixed 120 mm turns the carry into a descending sweep that ends
with the carried piece's base 124 mm up, 16 mm *below* the top of every king it
still has to cross.  Measured on franka/1d: 5 of 24 episodes ended
``board_disturbed`` during ``transfer``, all on the king.

:func:`place_goals` therefore takes the carry height as its ``approach_height``
and this constant is only the floor that :func:`carry_height` already enforces."""

PLACE_CLEARANCE = 0.004
"""Gap left under the piece when releasing, so it drops rather than being ground in [m]."""

GOAL_NAMES = ("pre_grasp", "grasp", "lift", "pre_place", "place")
"""The five keypoints the phase schedule interpolates between."""


def default_grasp_file(scale_tag: str = bl.PIECE_SCALE_TAG) -> Path:
    """Where ``lab/scripts/graspgen_chess_grasps.py`` writes its output."""
    return bl.GENERATED_ASSET_DIR / scale_tag / "grasps" / "chess_grasps.json"


def load_piece_geometry(scale_tag: str = bl.PIECE_SCALE_TAG) -> dict[str, dict[str, float]]:
    """``pieces.json``'s per-kind geometry: height, base diameter, mass, ...

    Read straight from the JSON rather than through :class:`assets.PieceAssets`,
    which would pull in newton and the collision meshes for numbers that are two
    floats per kind.
    """
    payload = json.loads((bl.GENERATED_ASSET_DIR / scale_tag / "pieces.json").read_text())
    return payload["pieces"]


##
# Quaternion and pose algebra (numpy, xyzw, batched over leading dimensions).
#
# warp has all of this, but the planner and the phase schedule run on the host over
# (world, candidate) arrays, and a round trip through warp per candidate would cost
# more than the arithmetic. Verified against wp.quat_from_matrix / wp.quat_slerp.
##


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of two xyzw quaternions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_inverse(quat: np.ndarray) -> np.ndarray:
    """Conjugate of a *unit* xyzw quaternion."""
    quat = np.asarray(quat, dtype=np.float64)
    return np.concatenate([-quat[..., :3], quat[..., 3:]], axis=-1)


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by an xyzw quaternion."""
    quat = np.asarray(quat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    axis, w = quat[..., :3], quat[..., 3:]
    return vec + 2.0 * np.cross(axis, np.cross(axis, vec) + w * vec)


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """xyzw quaternion(s) -> ``(..., 3, 3)`` rotation matrix."""
    quat = quat_normalize(quat)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrix -> xyzw quaternion(s).

    Shepperd's method: pick whichever of ``w, x, y, z`` is largest and divide by
    it, so the division is never by a small number.  Building the quaternion from
    the trace alone loses all precision for rotations near 180 degrees, which the
    grasp set is full of -- most of these grasps point the hand straight down.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    m = [[matrix[..., i, j] for j in range(3)] for i in range(3)]
    trace = m[0][0] + m[1][1] + m[2][2]

    candidates = np.stack([trace, m[0][0], m[1][1], m[2][2]], axis=-1)
    branch = np.argmax(candidates, axis=-1)

    quats = np.empty(matrix.shape[:-2] + (4, 4), dtype=np.float64)
    root = np.sqrt(np.maximum(1.0 + trace, 1e-12))
    quats[..., 0, :] = np.stack(
        [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1], root * root], axis=-1
    ) / (2.0 * root)[..., None]
    for axis in range(3):
        other = [(axis + 1) % 3, (axis + 2) % 3]
        root = np.sqrt(np.maximum(1.0 + m[axis][axis] - m[other[0]][other[0]] - m[other[1]][other[1]], 1e-12))
        values = [None, None, None, None]
        values[axis] = root * root
        values[other[0]] = m[other[0]][axis] + m[axis][other[0]]
        values[other[1]] = m[axis][other[1]] + m[other[1]][axis]
        values[3] = m[other[1]][other[0]] - m[other[0]][other[1]]
        quats[..., axis + 1, :] = np.stack(values, axis=-1) / (2.0 * root)[..., None]

    picked = np.take_along_axis(quats, branch[..., None, None], axis=-2)[..., 0, :]
    return quat_normalize(picked)


def pose_to_matrix(position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """``(..., 3)`` + ``(..., 4)`` xyzw -> ``(..., 4, 4)``."""
    position = np.asarray(position, dtype=np.float64)
    matrix = np.zeros(position.shape[:-1] + (4, 4), dtype=np.float64)
    matrix[..., :3, :3] = quat_to_matrix(quat)
    matrix[..., :3, 3] = position
    matrix[..., 3, 3] = 1.0
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(..., 4, 4)`` -> ``(..., 3)`` + ``(..., 4)`` xyzw."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[..., :3, 3], quat_from_matrix(matrix[..., :3, :3])


def rotation_z(angle: np.ndarray) -> np.ndarray:
    """``(...)`` angles -> ``(..., 4, 4)`` rotations about Z."""
    angle = np.asarray(angle, dtype=np.float64)
    matrices = np.zeros(angle.shape + (4, 4), dtype=np.float64)
    cos, sin = np.cos(angle), np.sin(angle)
    matrices[..., 0, 0] = cos
    matrices[..., 0, 1] = -sin
    matrices[..., 1, 0] = sin
    matrices[..., 1, 1] = cos
    matrices[..., 2, 2] = 1.0
    matrices[..., 3, 3] = 1.0
    return matrices


def yaw_quat_of(quat: np.ndarray) -> np.ndarray:
    """Drop roll and pitch, keep yaw. ``isaaclab.utils.math.yaw_quat``."""
    quat = np.asarray(quat, dtype=np.float64)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zeros = np.zeros_like(yaw)
    return quat_normalize(np.stack([zeros, zeros, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], axis=-1))


def quat_slerp(quat_a: np.ndarray, quat_b: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Shortest-arc geodesic interpolation between xyzw quaternions.

    Equal to the lab's ``quat_box_plus(a, tau * quat_box_minus(b, a))``: the world-
    frame geodesic ``exp(tau * log(b a^-1)) a`` and the body-frame one
    ``a (a^-1 b)^tau`` are the same rotation for unit quaternions.
    """
    quat_a = quat_normalize(quat_a)
    quat_b = quat_normalize(quat_b)
    tau = np.asarray(tau, dtype=np.float64)[..., None]

    dot = np.sum(quat_a * quat_b, axis=-1, keepdims=True)
    # q and -q are the same rotation; without this the interpolation takes the long
    # way round for more than half of the grasp set.
    quat_b = np.where(dot < 0.0, -quat_b, quat_b)
    dot = np.abs(dot)

    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_angle = np.sin(angle)
    # Fall back to a normalised lerp when the two are nearly parallel, where
    # sin(angle) underflows.
    parallel = sin_angle < 1e-6
    weight_a = np.where(parallel, 1.0 - tau, np.sin((1.0 - tau) * angle) / np.where(parallel, 1.0, sin_angle))
    weight_b = np.where(parallel, tau, np.sin(tau * angle) / np.where(parallel, 1.0, sin_angle))
    return quat_normalize(weight_a * quat_a + weight_b * quat_b)


def quat_error_magnitude(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    """Shortest rotation angle [rad] between two xyzw quaternions, in ``[0, pi]``."""
    relative = quat_mul(quat_normalize(quat_a), quat_inverse(quat_normalize(quat_b)))
    return 2.0 * np.arctan2(np.linalg.norm(relative[..., :3], axis=-1), np.abs(relative[..., 3]))


##
# Grasp library
##


def gripper_probe_points(spec: NewtonRobotSpec) -> np.ndarray:
    """Sample points on the two open fingers, in the ``ee_body`` frame. ``(24, 3)``.

    A slab per finger, running back from the TCP along the approach axis and offset
    to either side along the closing axis.  Crude, but it is only used to ask
    "would these fingers dip into a neighbouring piece", and it needs no per-arm
    mesh -- which matters because two of the four arms are URDF imports with no
    usable finger geometry to sample.
    """
    approach = np.asarray(spec.approach_axis, dtype=np.float64)
    closing = np.asarray(spec.closing_axis, dtype=np.float64)
    lateral = np.cross(approach, closing)
    tcp = np.asarray(spec.tcp_offset, dtype=np.float64)

    half = spec.max_opening / 2.0
    points = []
    for side in (1.0, -1.0):
        for offset in (half, half + FINGER_THICKNESS):
            for back in (0.0, 0.5 * FINGER_LENGTH, FINGER_LENGTH):
                for across in (-FINGER_THICKNESS, FINGER_THICKNESS):
                    points.append(tcp - approach * back + closing * (side * offset) + lateral * across)
    return np.stack(points)


class GraspLibrary:
    """GraspGen grasps per piece kind, retargeted to one arm's end-effector frame."""

    def __init__(self, grasp_file: str | Path, spec: NewtonRobotSpec, max_candidates: int = 12) -> None:
        grasp_file = Path(grasp_file)
        if not grasp_file.exists():
            raise FileNotFoundError(
                f"Missing GraspGen grasps at {grasp_file}. Regenerate them with GraspGen's own"
                f" interpreter (see lab/Readme.md):\n"
                f"    <graspgen-venv>/bin/python {bl.REPO_ROOT}/lab/scripts/graspgen_chess_grasps.py"
            )
        payload = json.loads(grasp_file.read_text())
        gripper = payload.get("gripper")
        if gripper != "franka_panda":
            raise ValueError(f"{grasp_file} was generated for gripper {gripper!r}, expected 'franka_panda'")

        self.path = grasp_file
        self.spec = spec
        self.piece_scale: float = payload["piece_scale"]
        self.gripper_depth: float = payload["gripper_depth"]
        self.max_candidates = max_candidates

        # The one place the arm enters: GraspGen's (approach +Z, closing +X, origin
        # at the gripper base) frame carried onto this arm's ee_body frame.
        self.convention_fix = spec.graspgen_to_ee(self.gripper_depth)

        self.hand_in_piece: dict[str, np.ndarray] = {}
        self.scores: dict[str, np.ndarray] = {}
        self.yaw_free: dict[str, bool] = {}
        for kind, result in payload["pieces"].items():
            records = result["grasps"][:max_candidates]
            matrices = np.asarray([record["matrix"] for record in records], dtype=np.float64).reshape(-1, 4, 4)
            self.hand_in_piece[kind] = matrices @ self.convention_fix
            self.scores[kind] = np.asarray([record["rank_score"] for record in records], dtype=np.float64)
            self.yaw_free[kind] = bool(result["yaw_free"])

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self.hand_in_piece)

    def candidates(self, kind: str, num_yaws: int) -> tuple[np.ndarray, np.ndarray]:
        """All ``(grasp x yaw)`` hand poses in the piece frame, plus their scores.

        The yaw rotation is applied on the **left**: the matrices live in the piece
        frame and every piece except the knight is a solid of revolution about its
        own ``+Z``, so spinning the piece frame is what generates equivalent grasps.
        Score index ordering is ``yaw * C + candidate``, matching the tiled scores.
        """
        matrices = self.hand_in_piece[kind]
        scores = self.scores[kind]
        if not self.yaw_free[kind]:
            return matrices, scores
        yaws = np.arange(num_yaws, dtype=np.float64) * (2.0 * math.pi / num_yaws)
        spun = rotation_z(yaws)[:, None] @ matrices[None, :]
        return spun.reshape(-1, 4, 4), np.tile(scores, num_yaws)


##
# Planning
##


def carry_height(max_piece_height: float, reach: float) -> float:
    """How high above the board the piece is carried [m].

    The carried piece hangs below the hand with its base at the lift height, so the
    carry has to clear the tallest piece still standing -- a fixed lift leaves a
    pawn's base level with the top of the king and drags it off the board.
    """
    wanted = max_piece_height + CARRY_CLEARANCE
    return max(MIN_LIFT_HEIGHT, min(wanted, CARRY_REACH_FRACTION * reach))


@dataclass(frozen=True)
class GraspPlan:
    """The winning candidate and the keypoints derived from it, all in world frame."""

    hand_pose_w: np.ndarray
    """4x4 pose of ``ee_body`` at the grasp."""

    goals: dict[str, tuple[np.ndarray, np.ndarray]]
    """``name -> (position, xyzw quaternion)`` for every entry of :data:`GOAL_NAMES`."""

    candidate: int
    score: float
    penetration: float
    """Worst-case depth [m] the open fingers dip into a neighbouring piece."""


class GraspPlanner:
    """Chooses the GraspGen candidate the current board leaves room for."""

    def __init__(
        self,
        spec: NewtonRobotSpec,
        library: GraspLibrary,
        piece_geometry: Mapping[str, Mapping[str, float]],
        num_yaws: int = 16,
    ) -> None:
        self.spec = spec
        self.library = library
        self.num_yaws = num_yaws
        self.probe_points = gripper_probe_points(spec)
        self.radius = {kind: float(entry["base_diameter"]) / 2.0 for kind, entry in piece_geometry.items()}
        self.height = {kind: float(entry["height"]) for kind, entry in piece_geometry.items()}
        # Cache: the candidate set depends only on (kind, num_yaws), and planning
        # happens once per world per episode, so this saves a 192x4x4 matmul chain
        # every reset.
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def candidates(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        if kind not in self._cache:
            self._cache[kind] = self.library.candidates(kind, self.num_yaws)
        return self._cache[kind]

    def approach_directions(self, hand_quat: np.ndarray) -> np.ndarray:
        """World approach direction(s) of the gripper at the given hand rotation(s).

        The Isaac Lab implementation uses the ``ee_body`` frame's local **+Z** here,
        which is the approach axis for three of the four arms but the *lateral* axis
        for the reBot (whose ``gripper_end`` reaches along its own +X).  On the
        reBot that backs the pre-grasp off sideways instead of along the approach,
        which is a documented latent bug there; rotating ``spec.approach_axis`` is
        the generalisation, and is bit-identical for franka, piper and yam.
        """
        return quat_rotate(hand_quat, np.asarray(self.spec.approach_axis, dtype=np.float64))

    def plan(
        self,
        kind: str,
        piece_pose_w: np.ndarray,
        other_centers: np.ndarray,
        other_kinds: Sequence[str],
        lift_height: float,
    ) -> GraspPlan:
        """Score every candidate against the board and derive the pick keypoints.

        *piece_pose_w* is the commanded piece's 4x4 world pose, *other_centers* the
        ``(O, 3)`` world base-plane positions of every other piece and *other_kinds*
        their kinds.  Nothing else on the scene is checked -- not the board, not the
        table, not the arm's own links -- exactly as in the Isaac Lab version.
        """
        candidates, scores = self.candidates(kind)
        hand_w = piece_pose_w[None] @ candidates  # (K, 4, 4)

        grasp_points = self.probe_points[None] @ np.swapaxes(hand_w[:, :3, :3], 1, 2) + hand_w[:, None, :3, 3]
        # The fingers have to reach the grasp as well as hold it, so the pre-grasp
        # cloud is scored too.
        standoff = self.approach_directions(quat_from_matrix(hand_w[:, :3, :3]))
        approach_points = grasp_points - standoff[:, None, :] * APPROACH_STANDOFF
        probe_world = np.concatenate([grasp_points, approach_points], axis=1)  # (K, 2P, 3)

        penetration = np.zeros(len(candidates))
        other_centers = np.asarray(other_centers, dtype=np.float64).reshape(-1, 3)
        if len(other_centers):
            radii = np.asarray([self.radius[k] for k in other_kinds], dtype=np.float64)
            tops = other_centers[:, 2] + np.asarray([self.height[k] for k in other_kinds], dtype=np.float64)
            # Each neighbour is an upright column of infinite depth: a probe is inside
            # it when it is both within the (inflated) radius and below the (inflated)
            # top, and the penalty is the smaller of the two margins.
            radial = np.linalg.norm(probe_world[:, :, None, :2] - other_centers[None, None, :, :2], axis=-1)
            inside_column = (radii[None, None, :] + CLEARANCE_MARGIN) - radial
            below_top = (tops[None, None, :] + CLEARANCE_MARGIN) - probe_world[:, :, None, 2]
            overlap = np.clip(np.minimum(inside_column, below_top), 0.0, None)
            penetration = overlap.max(axis=(1, 2))

        best = int(np.argmax(scores - COLLISION_WEIGHT * penetration))
        grasp_w = hand_w[best]
        grasp_pos, grasp_quat = matrix_to_pose(grasp_w)
        approach = self.approach_directions(grasp_quat)

        lift_pos = grasp_pos + np.array([0.0, 0.0, lift_height])
        goals = {
            "grasp": (grasp_pos, grasp_quat),
            "pre_grasp": (grasp_pos - approach * APPROACH_STANDOFF, grasp_quat),
            "lift": (lift_pos, grasp_quat),
            # Provisional: refreshed by place_goals() once the piece is actually held.
            "place": (lift_pos, grasp_quat),
            "pre_place": (lift_pos, grasp_quat),
        }
        return GraspPlan(
            hand_pose_w=grasp_w,
            goals=goals,
            candidate=best,
            score=float(scores[best]),
            penetration=float(penetration[best]),
        )


def place_goals(
    target_pos_w: np.ndarray,
    piece_pose_w: np.ndarray,
    hand_pose_w: np.ndarray,
    approach_height: float = PLACE_APPROACH_HEIGHT,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Freeze the hand-to-piece transform and turn the target square into hand poses.

    Returns ``(hand_in_piece, {"place": ..., "pre_place": ...})``.  The piece is put
    down upright, keeping the yaw it had when it was picked up, which is why the
    place pose cannot be derived from the grasp plan: how the piece actually ended
    up in the hand is only known once it is held.

    *approach_height* is where ``transfer`` ends and ``place`` begins; pass the
    episode's carry height so the carry stays level (see
    :data:`PLACE_APPROACH_HEIGHT`).
    """
    hand_in_piece = np.linalg.inv(piece_pose_w) @ hand_pose_w

    target = np.asarray(target_pos_w, dtype=np.float64) + np.array([0.0, 0.0, PLACE_CLEARANCE])
    upright = yaw_quat_of(quat_from_matrix(piece_pose_w[:3, :3]))
    place_hand_w = pose_to_matrix(target, upright) @ hand_in_piece

    place_pos, place_quat = matrix_to_pose(place_hand_w)
    return hand_in_piece, {
        "place": (place_pos, place_quat),
        "pre_place": (place_pos + np.array([0.0, 0.0, approach_height]), place_quat),
    }
