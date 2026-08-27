"""Pose algebra for the Genesis port, and the one place the two quaternion orders meet.

**The convention split, and why it is here rather than spread out.**

Three of the four things this port has to talk to disagree about quaternion order:

===================================  ==========
GraspGen JSON / Isaac Lab / warp     4x4 matrices, and ``(x, y, z, w)`` where a quaternion
                                     appears
Newton port (``robochess_newton``)   ``(x, y, z, w)``
Genesis 1.3.3                        ``(w, x, y, z)`` -- ``set_quat``, ``get_quat``,
                                     ``get_links_quat``, ``inverse_kinematics(quat=...)``,
                                     ``gs.morphs.*.quat``
===================================  ==========

Getting this wrong is not a crash, it is a *plausible* wrong answer: ``(0,0,0,1)`` and
``(1,0,0,0)`` are both unit quaternions and both read as "identity" to a careless eye, one
of them is a 180-degree turn, and the failure shows up several seconds later as a gripper
that descends beside the piece instead of onto it.

So the whole package works in **xyzw**, exactly like the Newton port it is a translation of
-- which keeps ``grasps.py`` and ``pick.py`` line-comparable with their Newton
counterparts, and keeps the GraspGen retargeting math untouched -- and every value that
crosses into or out of Genesis goes through :func:`to_gs_quat` / :func:`from_gs_quat`
*here*.  Nothing outside this module stores a wxyz quaternion, and nothing inside the
package passes a bare 4-vector to a Genesis call.

Everything else in this module is plain numpy over leading batch dimensions, ported from
``newton/robochess_newton/grasps.py``.  It runs on the host over ``(world, candidate)``
arrays a few hundred times per episode, not per physics substep.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "as_numpy",
    "from_gs_quat",
    "matrix_to_pose",
    "pose_to_matrix",
    "quat_error_magnitude",
    "quat_from_matrix",
    "quat_inverse",
    "quat_mul",
    "quat_normalize",
    "quat_rotate",
    "quat_slerp",
    "quat_to_matrix",
    "rotation_z",
    "to_gs_quat",
    "yaw_quat_of",
]


##
# The bridge. Two functions, both pure index permutations, both batched.
##


def to_gs_quat(quat_xyzw: np.ndarray | Sequence[float]) -> np.ndarray:
    """``(..., 4)`` xyzw -> ``(..., 4)`` wxyz, i.e. this package -> Genesis."""
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    return np.concatenate([quat[..., 3:4], quat[..., 0:3]], axis=-1)


def from_gs_quat(quat_wxyz: Any) -> np.ndarray:
    """``(..., 4)`` wxyz -> ``(..., 4)`` xyzw, i.e. Genesis -> this package.

    Accepts a torch tensor (which is what every Genesis getter returns) as well as
    anything numpy can view, so call sites do not each need their own ``.cpu().numpy()``.
    """
    quat = as_numpy(quat_wxyz)
    return np.concatenate([quat[..., 1:4], quat[..., 0:1]], axis=-1)


def as_numpy(value: Any) -> np.ndarray:
    """A float64 numpy view of a Genesis return value.

    Genesis returns ``torch.Tensor`` on the simulation device, and this package's
    schedule, planner and termination logic are all host numpy.  One helper rather than
    ``.detach().cpu().numpy()`` at forty call sites, and it degrades gracefully for the
    handful of Genesis getters that already hand back numpy.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


##
# Quaternion and pose algebra (numpy, xyzw, batched over leading dimensions).
#
# Ported verbatim from newton/robochess_newton/grasps.py, which validated it against
# wp.quat_from_matrix / wp.quat_slerp. Poses are 4x4 row-major homogeneous matrices,
# matching the GraspGen JSON.
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

    Shepperd's method: pick whichever of ``w, x, y, z`` is largest and divide by it, so the
    division is never by a small number.  Building the quaternion from the trace alone
    loses all precision for rotations near 180 degrees, which the grasp set is full of --
    most of these grasps point the hand straight down.
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

    Equal to the lab's ``quat_box_plus(a, tau * quat_box_minus(b, a))``: the world-frame
    geodesic ``exp(tau * log(b a^-1)) a`` and the body-frame one ``a (a^-1 b)^tau`` are the
    same rotation for unit quaternions.
    """
    quat_a = quat_normalize(quat_a)
    quat_b = quat_normalize(quat_b)
    tau = np.asarray(tau, dtype=np.float64)[..., None]

    dot = np.sum(quat_a * quat_b, axis=-1, keepdims=True)
    # q and -q are the same rotation; without this the interpolation takes the long way
    # round for more than half of the grasp set.
    quat_b = np.where(dot < 0.0, -quat_b, quat_b)
    dot = np.abs(dot)

    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_angle = np.sin(angle)
    # Fall back to a normalised lerp when the two are nearly parallel, where sin(angle)
    # underflows.
    parallel = sin_angle < 1e-6
    weight_a = np.where(parallel, 1.0 - tau, np.sin((1.0 - tau) * angle) / np.where(parallel, 1.0, sin_angle))
    weight_b = np.where(parallel, tau, np.sin(tau * angle) / np.where(parallel, 1.0, sin_angle))
    return quat_normalize(weight_a * quat_a + weight_b * quat_b)


def quat_error_magnitude(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    """Shortest rotation angle [rad] between two xyzw quaternions, in ``[0, pi]``."""
    relative = quat_mul(quat_normalize(quat_a), quat_inverse(quat_normalize(quat_b)))
    return 2.0 * np.arctan2(np.linalg.norm(relative[..., :3], axis=-1), np.abs(relative[..., 3]))
