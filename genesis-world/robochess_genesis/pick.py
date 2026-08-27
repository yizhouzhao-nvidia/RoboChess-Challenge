"""Scripted pick-and-place of a chess piece, batched over the worlds of a scene.

The Genesis counterpart of ``ChessPickPolicy`` in
``lab/scripts/generate_chess_pick_demos.py``, translated from
``robochess_newton.pick.ChessPickTask``.  Each episode asks one question -- move piece *i*
to destination *j* -- and answers it with a nine-phase schedule (``pre_grasp, descend,
close, lift, transfer, place, release, retreat, settle``) whose keypoints come from the
GraspGen grasp chosen by :class:`~robochess_genesis.grasps.GraspPlanner`.

Three things about the schedule are not cosmetic and are reproduced exactly:

* **Each phase is a time interpolation *and* an arrival test.**  Inverse kinematics takes a
  bounded step per control tick, so a purely time-triggered schedule closes the fingers
  wherever the arm happens to be -- tens of millimetres short of the grasp, which slides the
  piece out.  A phase therefore holds at its goal until the pose error is inside its own
  tolerance, with a per-phase deadline so a bad IK solution cannot stall the episode.
* **The next leg starts from the previous *command*, not the measured pose.**  That keeps
  the commanded trajectory continuous instead of baking each leg's tracking error into the
  start of the next one.
* **The place pose is derived from the measured hand-to-piece transform**, refreshed at the
  end of ``close``, ``lift`` and ``transfer`` and only while the piece is actually held.
  Deriving it from a failed grasp aims the arm at a pose metres away and turns a quiet
  failure into a thrashing one.

The two deliberate departures from the Isaac Lab schedule that the Newton port introduced
are kept, for the same reason -- a solver-side IK that tracks its command tightly turns two
of its latent quirks into real failures:

* the leg's last commanded pose is the goal, not ``(N-1)/N`` of the way to it
  (:meth:`ChessPickTask._compute`);
* ``transfer`` ends at the carry height rather than a fixed 120 mm, so the carry is level
  instead of a descending sweep across the pieces still standing
  (:data:`~robochess_genesis.grasps.PLACE_APPROACH_HEIGHT`).

## What is Genesis-specific

**IK.** ``RigidEntity.inverse_kinematics`` is a damped-least-squares solve that, by default,
does up to ``max_samples=50`` restarts -- and a restart *uniformly resamples every joint
inside its limits* before descending again.  That is the right behaviour for "find me a
posture reaching this pose" and the wrong one for a servo loop: a leg whose target drifts
briefly out of reach would come back with a completely different arm configuration, and the
next control tick would command a violent reconfiguration through the board.  This task
therefore runs ``max_samples=1``, which makes the call a plain warm-started descent from the
previous solution -- the same thing ``newton.ik``'s Levenberg-Marquardt step does.  It is
also no slower: the call cost is launch-bound at ~25 ms whatever the iteration budget
(measured 27.2 ms at 50x20 against 27.8 ms at 1x24).

**IK failure is silent.**  Genesis reports non-convergence only through
``return_error=True``, and a sentinel error of exactly 1e4 means "no sample ever improved on
the seed", in which case the returned qpos is uninitialised memory.  Both are checked here;
see :meth:`ChessPickTask._solve_ik`.

**Batching.**  A batched Genesis env is a *copy* of the scene, so when every world is
identical there is one arm entity and one IK call covers all of them.  When the worlds
differ they share env 0 and each has its own entity, so there is one call per world.
:attr:`ChessPickTask.ik_groups` holds whichever it is, and nothing else in the file cares.

**Gains are set at build time.**  Genesis exposes ``set_dofs_kp`` / ``set_dofs_armature``
only on a built solver, so the pick gains go through ``ChessScene.finalize(...)`` rather
than through a Newton-style pre-finalize edit; :func:`pick_gain_kwargs` packages them.

The schedule itself is numpy vectorised over worlds on the host, one array op per quantity
rather than a Python loop, and runs at 30 Hz.  The only per-world Python is grasp planning,
which happens once per episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# grasps first: it imports board_layout, which installs PXR_WORK_THREAD_LIMIT.
from . import grasps as G
from . import gsmath as M
from .robots import GenesisRobotSpec, get_spec
from .scene import ChessScene

import numpy as np

__all__ = [
    "ARM_JOINT_JITTER",
    "CONTROL_DT",
    "DISTURB_MAX_TILT_DEG",
    "EPISODE_LENGTH_S",
    "GRASP_MAX_FRACTION",
    "GRASP_MAX_TCP_DISTANCE",
    "GRASP_MIN_FRACTION",
    "GRIPPER_TRAVEL_TIME",
    "IK_DAMPING",
    "IK_ITERATIONS",
    "IK_MAX_STEP",
    "MIN_PIECE_HEIGHT",
    "PHASES",
    "PICK_ARM_ARMATURE",
    "PICK_ARM_KP",
    "PICK_DAMPING_RATIO",
    "PICK_GRIPPER_ARMATURE",
    "PICK_GRIPPER_KP",
    "PIECE_XY_JITTER",
    "PIECE_YAW_JITTER",
    "PLACE_MAX_SPEED",
    "PLACE_MAX_TILT_DEG",
    "PLACE_XY_TOLERANCE",
    "PLACE_Z_TOLERANCE",
    "ChessPickTask",
    "EpisodeResult",
    "Phase",
    "pick_gain_kwargs",
    "resolve_gripper_kp",
]

##
# Rates. The Isaac Lab task runs sim at 1/120 with decimation 4; this scene runs at 1/240
# like the Newton port, so a control tick is two 60 Hz frames instead of one.
##

CONTROL_DT = 1.0 / 30.0
"""Control period [s]. One IK solve and one schedule tick per period."""

EPISODE_LENGTH_S = 24.0
"""Episode budget [s]. The nominal schedule is 10.5 s and its worst case 20.5 s.

Every phase rolls over at its own deadline and a world is judged as soon as the schedule
ends, so at this budget nothing reaches it: ``timed_out`` is the guard against a shortened
``episode_length_s``, not the usual failure label."""

IK_ITERATIONS = 24
"""Damped-least-squares iterations per control tick, warm-started from the previous solve.

Matched to the Newton port's ``IK_ITERATIONS``. Genesis' call cost is launch-bound, so this
is free to be generous: 12, 24 and 50 iterations all measured within 1 ms of each other and
all converged the same warm-started step to 2 um."""

IK_DAMPING = 0.01
"""Levenberg-Marquardt damping. Genesis' default; it enters as ``damping**2``."""

IK_MAX_STEP = 0.5
"""Per-iteration clamp on each joint's step [rad or m]. Genesis' default."""

GRIPPER_TRAVEL_TIME = 0.5
"""How long the fingers take to cross their full stroke [s].

Isaac Lab's ``BinaryJointPositionAction`` steps the finger targets from open to closed in
one tick, and its PhysX implicit actuators absorb that.  A position-servo actuator does not:
the step is a 40 mm command error against the picking stiffness, i.e. hundreds of newtons on
the first substep, and a 25 g pawn is flicked out of the hand before the jaws touch it.
Ramping the command over 0.5 s -- shorter than the 0.9 s ``close`` phase, so the schedule is
unchanged -- keeps the jaws below the piece's tipping force.  The same value, and the same
reason, as the Newton port."""

##
# Drive tuning for picking. The scene ships 2000/100 on every DOF, which holds any of these
# arms within a few milliradians of a *static* pose once gravity is compensated -- enough to
# visualise a board. Closing a gripper on a 25 g piece is a different load case.
##

PICK_DAMPING_RATIO = 1.0 / 20.0
"""Position-drive damping as a fraction of the stiffness, on the arm and the gripper.

The scene ships 2000/100, the same ratio, so scaling the damping with whatever stiffness is
requested keeps a stiffer drive from being left underdamped."""

PICK_ARM_KP = 4000.0
"""Position-drive stiffness on the arm DOFs.

The open Franka hand is 106 mm across the fingers' outer faces and a board square is 84 mm,
so descending onto a piece grazes its neighbours; a soft arm stops short of the grasp and
the fingers close on air.  4000 is also where Isaac Lab's own config sits (4500 on the
Franka's forearm, 4000 on piper/rebot), and it is what the Newton port converged on."""

PICK_GRIPPER_KP = 300.0
"""Position-drive stiffness on the gripper DOFs -- deliberately *not* the arm's, and
deliberately **half the Newton port's 600**.

A position servo applies ``kp`` times the position error and the close command is the jaws'
hard stop, so once a jaw touches the piece the command keeps running past it and the squeeze
grows without limit.  Every piece here is a *tapered* shaft, so an over-squeezed grip does
not merely crush -- it extrudes the piece out of the jaws like a bar of soap, and the taller
the piece the worse the resulting swing.

The Newton port converged on 600 against a MuJoCo contact solve.  Measured here, franka,
16 episodes at seed 0, three runs per value, on the two boards that carry kings:

===== =============== =============== =========================================
kp    ``1d``          ``4x4``         where it goes wrong
===== =============== =============== =========================================
300   13, 14, 14      15, 15, 15      --
600   12, 12, 13      13, 13, 13      king 4-5/6 and 1/3
1200  13, 14, 13      14, 14, 14      knight 0-1/3
2000  10              --              rook 3/7 -- visibly extruded
===== =============== =============== =========================================

300 is better than 600 on both boards in all three repeats, and it is the only value that
is not worst-in-class on some piece kind.  The king is what separates them: it is the
heaviest piece (103 g) pinched at the narrowest neck (8.8 mm) at the greatest height
(126.8 mm of a 140.5 mm piece), so it is the one that a squeeze-out turns into a pendulum.
Note the ordering is *not* monotone -- 1200 recovers the king and loses the knight -- so
this is a measured optimum, not a trend to extrapolate."""

PICK_ARM_ARMATURE = 0.3
"""Rotor inertia added to every arm DOF [kg m^2].

Genesis already defaults URDF and MJCF joints to 0.1 (``gs.morphs.URDF.default_armature``),
so unlike Newton -- whose importers leave it at 0, which is why the Newton port had to
discover armature the hard way -- picking here does not *depend* on setting it.  It is still
set explicitly, to the Newton port's value, so that all four arms agree whatever their asset
authors (the YAM MJCF authors 0.032 on the shoulder and 0.0018 on the wrist) and so the two
ports are driving comparable systems."""

PICK_GRIPPER_ARMATURE = 0.15
"""Rotor inertia added to every gripper DOF [kg m^2]. The Newton port's value."""


def resolve_gripper_kp(spec: GenesisRobotSpec | str | None, override: float | None = None) -> float:
    """The gripper stiffness to use: the override, else the arm's own, else the default."""
    if override is not None:
        return override
    if spec is None:
        return PICK_GRIPPER_KP
    value = get_spec(spec).pick_gripper_kp
    return PICK_GRIPPER_KP if value is None else value


def pick_gain_kwargs(
    arm_kp: float = PICK_ARM_KP,
    gripper_kp: float | None = None,
    arm_armature: float = PICK_ARM_ARMATURE,
    gripper_armature: float = PICK_GRIPPER_ARMATURE,
    spec: GenesisRobotSpec | str | None = None,
) -> dict[str, float]:
    """The ``ChessScene.finalize(**kwargs)`` a picking run wants.

    A function rather than a constant dict for two reasons: the damping is derived from the
    stiffness (:data:`PICK_DAMPING_RATIO`), so a caller that raises ``--arm-ke`` does not
    silently get an underdamped drive; and ``gripper_kp`` defaults *per arm* through
    :func:`resolve_gripper_kp` rather than to one number for everybody.
    """
    gripper_kp = resolve_gripper_kp(spec, gripper_kp)
    return {
        "arm_kp": arm_kp,
        "arm_kd": arm_kp * PICK_DAMPING_RATIO,
        "gripper_kp": gripper_kp,
        "gripper_kd": gripper_kp * PICK_DAMPING_RATIO,
        "arm_armature": arm_armature,
        "gripper_armature": gripper_armature,
    }


##
# Reset randomisation, from mdp/events.py.
##

PIECE_XY_JITTER = 0.004
"""Uniform +-XY offset applied to every piece at reset [m]."""

PIECE_YAW_JITTER = 0.35
"""Uniform +-yaw applied to every piece at reset [rad]."""

ARM_JOINT_JITTER = 0.02
"""Uniform +-offset applied to the arm joints at reset [rad].

Applied to the arm DOFs only.  Isaac Lab's ``reset_joints_by_offset`` also hits the finger
joints, which moves the gripper's open fraction by up to half its range and makes the
``piece_grasped`` predicate fire on an empty hand at reset; that is a quirk of the lab
config, not part of the task."""

##
# Success / termination, from mdp/terminations.py and mdp/observations.py.
##

GRASP_MAX_TCP_DISTANCE = 0.18
"""How far the TCP may be from the piece origin and still count as holding it [m].

Wide enough to cover the knight, which GraspGen grips off-axis by the head."""

GRASP_MIN_FRACTION = 0.04
GRASP_MAX_FRACTION = 0.94
"""Gripper-opening fraction bracket that counts as pinched.

Averaged over the fingers, never tested per finger: the fingers are independently actuated
and close asymmetrically on an off-centre grasp."""

PLACE_XY_TOLERANCE = 0.02
PLACE_Z_TOLERANCE = 0.01
PLACE_MAX_SPEED = 0.05
PLACE_MAX_TILT_DEG = 25.0
DISTURB_MAX_TILT_DEG = 30.0
"""Tilt beyond which a piece *other* than the commanded one counts as knocked over."""

MIN_PIECE_HEIGHT = 0.5
"""A piece below this world height has left the table [m] (the table top is at 0.77)."""

IK_ERROR_SENTINEL = 1e4
"""Genesis seeds its best-error accumulator to this; a returned error at or above it means
no sample ever improved on the seed and the returned qpos is uninitialised memory."""


@dataclass(frozen=True)
class Phase:
    """One leg of the pick-and-place: interpolate to ``goal``, then wait for arrival."""

    name: str
    duration: float
    """Interpolation time [s]; also the minimum time the leg takes."""

    gripper_open: bool
    goal: str
    """Which entry of :data:`~robochess_genesis.grasps.GOAL_NAMES` this leg drives to."""

    pos_tolerance: float = 0.004
    rot_tolerance: float = 0.06
    settle_timeout: float = 1.5
    """Extra time [s] the leg may spend waiting to arrive before it gives up and rolls over
    anyway."""


PHASES: tuple[Phase, ...] = (
    Phase("pre_grasp", 1.8, True, "pre_grasp", pos_tolerance=0.006),
    Phase("descend", 1.0, True, "grasp", pos_tolerance=0.003, rot_tolerance=0.04, settle_timeout=2.0),
    Phase("close", 0.9, False, "grasp", settle_timeout=0.0),
    Phase("lift", 1.0, False, "lift", pos_tolerance=0.008),
    Phase("transfer", 2.2, False, "pre_place", pos_tolerance=0.008),
    Phase("place", 1.2, False, "place", pos_tolerance=0.003, rot_tolerance=0.04, settle_timeout=2.0),
    Phase("release", 0.6, True, "place", settle_timeout=0.0),
    Phase("retreat", 1.0, True, "pre_place", pos_tolerance=0.02),
    Phase("settle", 0.8, True, "pre_place", settle_timeout=0.0),
)

CAPTURE_AFTER = ("close", "lift", "transfer")
"""Phases after which the hand-to-piece transform is re-measured.

Three points rather than one: the fingers may still be travelling when ``close`` ends, and
the piece can settle or slip during the lift and the carry."""

_PHASE_BY_NAME = {phase.name: index for index, phase in enumerate(PHASES)}
_GOAL_INDEX = {name: index for index, name in enumerate(G.GOAL_NAMES)}


@dataclass
class EpisodeResult:
    """What one world did with one move request."""

    world: int
    episode: int
    kind: str
    piece: str
    target: tuple[float, float, float]
    outcome: str
    """``success``, or the reason it was not.

    ``piece_off_board`` and ``board_disturbed`` are the two board-wrecking terminations;
    ``timed_out`` means the step budget ran out with the schedule still running.  A schedule
    that ran to the end without placing is labelled with the first clause of the success
    predicate that rejected it -- ``missed_target``, ``off_surface``, ``piece_tipped``,
    ``not_released`` or ``still_moving`` -- rather than being lumped in with the timeouts."""

    steps: int
    phase: str
    """Phase the world was in when the outcome was latched."""

    grasp_score: float
    grasp_penetration: float
    place_error: float
    """Planar distance from the piece to its destination when the outcome fired [m]."""

    @property
    def success(self) -> bool:
        return self.outcome == "success"


@dataclass
class _IKGroup:
    """One arm entity plus the worlds it drives.

    In a batched scene there is exactly one of these and it covers every world; when the
    worlds differ they share env 0 and each gets its own group.  ``env_rows`` is the
    ``envs_idx`` for the Genesis call and ``worlds`` the parallel list of world indices, so
    ``group.worlds[k]`` is solved in ``group.env_rows[k]``.
    """

    entity: Any
    link: Any
    dofs_local: list[int]
    worlds: list[int]
    env_rows: list[int]
    q_warm: np.ndarray
    """``(len(worlds), n_qs)`` warm start, updated in place every tick."""


class ChessPickTask:
    """Runs the nine-phase pick schedule in every world of a :class:`ChessScene`.

    The scene must be finalized and every world must carry the *same* arm (the grasp
    library and the probe geometry are per-arm); scenarios may differ.
    """

    def __init__(
        self,
        scene: ChessScene,
        *,
        grasp_file: str | Path | None = None,
        num_grasp_candidates: int = 12,
        num_yaw_candidates: int = 16,
        seed: int = 0,
        balance_kinds: bool = False,
        control_dt: float = CONTROL_DT,
        ik_iterations: int = IK_ITERATIONS,
        episode_length_s: float = EPISODE_LENGTH_S,
        debug: bool = False,
    ) -> None:
        if not scene.built:
            raise RuntimeError("ChessScene.finalize() must be called before ChessPickTask")
        specs = {world.spec.key for world in scene.worlds}
        if len(specs) != 1:
            raise ValueError(f"every world must use the same arm; got {sorted(specs)}")

        self.scene = scene
        self.spec: GenesisRobotSpec = get_spec(next(iter(specs)))
        if not self.spec.has_gripper:
            raise ValueError(f"{self.spec.key} has no gripper; picking needs a parallel jaw")

        self.world_count = scene.world_count
        self.debug = debug
        self.balance_kinds = balance_kinds
        self.rng = np.random.default_rng(seed)

        self.control_dt = control_dt
        self.frames_per_tick = int(round(control_dt * scene.fps))
        if abs(self.frames_per_tick / scene.fps - control_dt) > 1e-9:
            raise ValueError(f"control_dt {control_dt} is not a whole number of {scene.fps} Hz frames")
        self.max_steps = max(1, round(episode_length_s / control_dt))
        self.ik_iterations = ik_iterations
        self._tick_ik_missed = False
        self.ik_failures = 0
        """Control ticks on which **at least one** world's IK finished more than 1 mm from its
        commanded position. Counted per tick rather than per world, so with 8 worlds a 13 %
        tick rate is a ~1.7 % world-tick rate.

        Reported, not fatal. A leg whose target is briefly outside the workspace is a normal
        event -- the schedule commands a straight line between keypoints and does not check
        reachability -- and tracking as close as possible is the right response, which is also
        what the Newton port's Levenberg-Marquardt step does. What is *not* normal is this
        number approaching the tick count, which would mean the arm is not following at all."""

        self.library = G.GraspLibrary(
            grasp_file or G.default_grasp_file(), self.spec, max_candidates=num_grasp_candidates
        )
        self.piece_geometry = G.load_piece_geometry()
        self.planner = G.GraspPlanner(self.spec, self.library, self.piece_geometry, num_yaws=num_yaw_candidates)

        self._setup_indices()
        self._setup_schedule()
        self._setup_ik()

        self.episode_index = -1
        self.abandoned = 0
        """Worlds dropped unscored because the caller stopped stepping mid-schedule."""
        self.attempted_kinds: dict[str, int] = {kind: 0 for kind in self._all_kinds}
        self._preferred_kind: list[str | None] = [None] * self.world_count
        self.results: list[EpisodeResult] = []

    ##
    # Setup
    ##

    def _setup_indices(self) -> None:
        """Cache every scene slot the tick loop needs, as numpy arrays.

        The scene reads its whole watch list in one call per quantity, so what this needs
        is the ``(env, slot)`` pair for each world's hand and for every piece.  Pieces are
        ragged across worlds (a 1d board has 6, an 8x8 has 32), so they live in one flat
        array plus a world id, which keeps every per-piece test a single vectorised op --
        the same shape the Newton port uses.
        """
        worlds = self.scene.worlds
        spec = self.spec

        self._world_env = np.array([world.env for world in worlds], dtype=np.int64)
        self._ee_slot = np.array([world.ee_slot for world in worlds], dtype=np.int64)
        self._arm_dof_slots = np.array([world.arm_dof_slots for world in worlds], dtype=np.int64)
        self._grip_dof_slots = np.array([world.gripper_dof_slots for world in worlds], dtype=np.int64)

        self._grip_open = np.asarray(spec.gripper_open, dtype=np.float64)
        self._grip_close = np.asarray(spec.gripper_close, dtype=np.float64)
        span = self._grip_open - self._grip_close
        self._grip_valid = np.abs(span) > 1e-9
        if not self._grip_valid.any():
            raise ValueError(f"{spec.key}: open and close commands are identical, cannot detect a grasp")
        self._grip_span = np.where(self._grip_valid, span, 1.0)

        slots, envs, world_of, offsets, q_slots = [], [], [], [], []
        for index, world in enumerate(worlds):
            offsets.append(len(slots))
            slots.extend(world.piece_slots)
            q_slots.extend(world.piece_q_slots)
            envs.extend([world.env] * world.piece_count)
            world_of.extend([index] * world.piece_count)
        self._piece_slot = np.array(slots, dtype=np.int64)
        self._piece_q_slot = np.array(q_slots, dtype=np.int64)
        self._piece_env = np.array(envs, dtype=np.int64)
        self._piece_world = np.array(world_of, dtype=np.int64)
        self._piece_offset = np.array(offsets, dtype=np.int64)
        # Zero in the batched mode; the side-by-side offset when the worlds differ. Used by
        # the "left the table" test, which is the one termination expressed relative to the
        # world's own floor rather than in absolute coordinates.
        self._piece_origin_z = np.array([worlds[w].origin[2] for w in world_of])

        self._movable = [world.reachable_piece_indices() for world in worlds]
        self._targets = [np.asarray(world.target_positions(), dtype=np.float64) for world in worlds]
        for index, (movable, targets) in enumerate(zip(self._movable, self._targets)):
            if not movable:
                raise ValueError(
                    f"world {index}: no piece of the {worlds[index].layout.scenario} board is within "
                    f"{spec.key}'s {spec.reach:.2f} m reach"
                )
            if not len(targets):
                raise ValueError(
                    f"world {index}: no destination of the {worlds[index].layout.scenario} board is "
                    f"both within reach and on the table"
                )

        self._all_kinds = sorted({kind for world in worlds for kind in world.piece_kinds})
        self.lift_height = np.array(
            [
                G.carry_height(max(self.piece_geometry[k]["height"] for k in world.piece_kinds), spec.reach)
                for world in worlds
            ]
        )

        # The pristine spawn state: the arm at home, every piece on its square. Reset
        # perturbs a copy of these rather than re-deriving them.
        self._default_dofs = self.scene.home_dofs
        self._default_piece_qs = self.scene.spawn_piece_qs

    def _setup_schedule(self) -> None:
        step_dt = self.control_dt
        self._phase_steps = np.array([max(1, round(p.duration / step_dt)) for p in PHASES], dtype=np.int64)
        self._phase_deadline = self._phase_steps + np.array(
            [max(0, round(p.settle_timeout / step_dt)) for p in PHASES], dtype=np.int64
        )
        self._pos_tolerance = np.array([p.pos_tolerance for p in PHASES])
        self._rot_tolerance = np.array([p.rot_tolerance for p in PHASES])
        self._phase_goal = np.array([_GOAL_INDEX[p.goal] for p in PHASES], dtype=np.int64)
        self._phase_gripper_open = np.array([p.gripper_open for p in PHASES], dtype=bool)

        worlds = self.world_count
        self.phase_index = np.zeros(worlds, dtype=np.int64)
        self.phase_step = np.zeros(worlds, dtype=np.int64)
        self._goal_pos = np.zeros((len(G.GOAL_NAMES), worlds, 3))
        self._goal_quat = np.zeros((len(G.GOAL_NAMES), worlds, 4))
        self._goal_quat[..., 3] = 1.0
        self._start_pos = np.zeros((worlds, 3))
        self._start_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (worlds, 1))
        self._command_pos = np.zeros((worlds, 3))
        self._command_quat = self._start_quat.copy()
        self._grip_command = np.ones(worlds)
        """Commanded jaw opening, 1 = open, 0 = closed; rate limited, see
        :data:`GRIPPER_TRAVEL_TIME`."""
        self._grip_rate = self.control_dt / GRIPPER_TRAVEL_TIME

        self.piece_index = np.zeros(worlds, dtype=np.int64)
        self.target_index = np.zeros(worlds, dtype=np.int64)
        self.target_pos_w = np.zeros((worlds, 3))
        self.planned_kind = [""] * worlds
        self._plan_score = np.zeros(worlds)
        self._plan_penetration = np.zeros(worlds)

        self.done = np.zeros(worlds, dtype=bool)
        self.outcome: list[str] = [""] * worlds
        self._outcome_step = np.zeros(worlds, dtype=np.int64)
        self._outcome_phase = [""] * worlds
        self._outcome_place_error = np.zeros(worlds)
        self.control_step_index = 0

        # The DOF target array is seeded from the home posture once and only the arm and
        # gripper columns are ever written, so the rest never changes and does not have to
        # be read back each tick.
        self._target_dofs = self.scene.home_dofs

    def _setup_ik(self) -> None:
        """Group the worlds by the arm entity that drives them; see :class:`_IKGroup`."""
        groups: dict[int, _IKGroup] = {}
        for world in self.scene.worlds:
            handle = world.robot
            key = id(handle.entity)
            if key not in groups:
                groups[key] = _IKGroup(
                    entity=handle.entity,
                    link=handle.entity.links[handle.ee_link_local],
                    dofs_local=list(handle.arm_dofs_local),
                    worlds=[],
                    env_rows=[],
                    q_warm=np.zeros((0, handle.entity.n_qs)),
                )
            groups[key].worlds.append(world.index)
            groups[key].env_rows.append(world.env)
        for group in groups.values():
            # Genesis' IK returns and takes a *q*-space vector (n_qs), while dofs_idx_local
            # indexes DOF space. The two coincide only for a chain of 1-DOF joints, which is
            # what a fixed-base arm is -- a floating base would add a 7-coordinate free joint
            # against 6 DOFs and every index below would be off by one from the root down.
            if group.entity.n_qs != group.entity.n_dofs:
                raise RuntimeError(
                    f"{self.spec.key}: {group.entity.n_qs} coordinates for {group.entity.n_dofs} DOFs; "
                    "ChessPickTask assumes a fixed-base chain of 1-DOF joints"
                )
            group.q_warm = np.zeros((len(group.worlds), group.entity.n_qs), dtype=np.float32)
        self.ik_groups: tuple[_IKGroup, ...] = tuple(groups.values())

    ##
    # Episode lifecycle
    ##

    def reset_episode(self) -> None:
        """Randomise the scene, sample a move per world and plan its grasp.

        Every world resets together.  Isaac Lab auto-resets each environment the moment it
        terminates; here the caller reads the outcomes first, which makes the per-episode
        accounting exact and costs only the worlds that finish early idling until the
        slowest one is done.
        """
        scene = self.scene

        dofs = self._default_dofs.copy()
        jitter = self.rng.uniform(-ARM_JOINT_JITTER, ARM_JOINT_JITTER, size=self._arm_dof_slots.shape)
        dofs[self._world_env[:, None], self._arm_dof_slots] += jitter
        dofs[self._world_env[:, None], self._grip_dof_slots] = self._grip_open[None, :]

        # Free-joint coordinates, in the solver's own layout: xyz then a *wxyz* quaternion.
        # This is the one place in the package that handles wxyz outside gsmath, and it does
        # so by converting to xyzw for the multiply and straight back.
        piece_qs = self._default_piece_qs.copy()
        envs = self._piece_env
        base = self._piece_q_slot
        count = len(base)
        offsets = self.rng.uniform(-PIECE_XY_JITTER, PIECE_XY_JITTER, size=(count, 2))
        piece_qs[envs, base] += offsets[:, 0]
        piece_qs[envs, base + 1] += offsets[:, 1]
        yaw = self.rng.uniform(-PIECE_YAW_JITTER, PIECE_YAW_JITTER, size=count)
        spin = np.stack([np.zeros(count), np.zeros(count), np.sin(yaw / 2.0), np.cos(yaw / 2.0)], axis=-1)
        current = M.from_gs_quat(np.stack([piece_qs[envs, base + 3 + i] for i in range(4)], axis=-1))
        spun = M.to_gs_quat(M.quat_mul(current, spin))
        for i in range(4):
            piece_qs[envs, base + 3 + i] = spun[:, i]

        scene.write_dof_positions(dofs)
        scene.write_piece_qs(piece_qs)
        scene.zero_velocities()
        self._target_dofs = dofs.copy()
        scene.write_dof_targets(self._target_dofs)
        self._refresh_state()

        self.episode_index += 1
        self.phase_index[:] = 0
        self.phase_step[:] = 0
        self.done[:] = False
        self.outcome = [""] * self.world_count
        self._outcome_step[:] = 0
        self._outcome_phase = [""] * self.world_count
        self._outcome_place_error[:] = np.nan
        self.control_step_index = 0

        hand_pos, hand_quat = self._hand_pose()
        self._start_pos[:] = hand_pos
        self._start_quat[:] = hand_quat
        self._command_pos[:] = hand_pos
        self._command_quat[:] = hand_quat
        self._grip_command[:] = 1.0

        # Warm-start the IK from the arm's actual (jittered) posture, so the first command
        # -- the measured hand pose -- is already a solution.
        for group in self.ik_groups:
            q = M.as_numpy(group.entity.get_qpos(envs_idx=group.env_rows))
            group.q_warm = np.ascontiguousarray(q.reshape(len(group.worlds), -1), dtype=np.float32)

        for world in range(self.world_count):
            self._sample_command(world)
            self._plan_grasp(world)

    @property
    def episode_finished(self) -> bool:
        return bool(self.done.all()) or self.control_step_index >= self.max_steps

    def collect_results(self) -> list[EpisodeResult]:
        """Score the episode, one result per *finished* world; call once ``episode_finished``.

        A world that ran its episode out always has an outcome latched by
        :meth:`_evaluate_terminations` -- a termination, the placement verdict at the end of
        the schedule, or ``timed_out`` at the step budget.  So an unlatched world can only
        mean the caller stopped stepping early (a frame budget), and those are **not**
        returned: scoring a world still in ``descend`` would count it as a failed attempt and
        pull the reported success rate down with an episode that never got the chance to
        finish.
        """
        results = []
        for world in np.flatnonzero(self.done).tolist():
            index = int(self.piece_index[world])
            world_info = self.scene.worlds[world]
            results.append(
                EpisodeResult(
                    world=world,
                    episode=self.episode_index,
                    kind=self.planned_kind[world],
                    piece=world_info.piece_names[index],
                    target=tuple(self.target_pos_w[world]),
                    outcome=self.outcome[world],
                    steps=int(self._outcome_step[world]),
                    phase=self._outcome_phase[world],
                    grasp_score=float(self._plan_score[world]),
                    grasp_penetration=float(self._plan_penetration[world]),
                    place_error=float(self._outcome_place_error[world]),
                )
            )
            self.attempted_kinds[self.planned_kind[world]] += 1

        unfinished = self.world_count - len(results)
        if unfinished:
            self.abandoned += unfinished
            print(
                f"[WARN] {unfinished} of {self.world_count} worlds were still mid-schedule when"
                f" stepping stopped at control tick {self.control_step_index}; not scored"
            )
        self.results.extend(results)
        if self.balance_kinds:
            self._assign_preferred_kinds()
        return results

    ##
    # Per-tick
    ##

    def step(self, on_frame: Callable[[], None] | None = None) -> None:
        """One 30 Hz control tick: command, IK, physics, arrival test, terminations.

        *on_frame* is called after every simulated 60 Hz frame, which is where a capture or a
        frame budget hooks in.  It is **not** what decides whether the frame is rendered --
        ``ChessScene.renders`` is, from the scene's own camera and viewer -- so a caller may
        pass a hook that only counts frames without turning the render cost back on.
        """
        command_pos, command_quat, grip, interp_steps = self._compute()
        self._advance(command_pos, command_quat, interp_steps)
        self._drive(command_pos, command_quat, grip)

        for _ in range(self.frames_per_tick):
            self.scene.step()
            if on_frame is not None:
                on_frame()

        self.control_step_index += 1
        self._refresh_state()
        self._evaluate_terminations()
        if self.debug:
            print(self.report(0))

    def _compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate from the previous command toward the current phase's goal.

        The one place this deliberately departs from ``ChessPickPolicy``.  Isaac Lab computes
        ``tau = phase_step / N`` *before* incrementing ``phase_step``, so the first rollover
        opportunity comes while ``tau = (N-1)/N`` and the exact goal is only ever commanded
        by a leg that failed to arrive on time.  Under a differential-IK action term that is
        invisible, because the arm never arrives on the first opportunity.  A solver-side IK
        tracks to a fraction of a millimetre, so here it fires every time and every leg would
        stop one step short -- 3 mm on ``descend``, which is half the bishop's 6.9 mm neck.
        Counting from ``phase_step + 1`` makes the last commanded pose the goal itself at no
        cost in schedule length.
        """
        index = np.minimum(self.phase_index, len(PHASES) - 1)
        interp = self._phase_steps[index]
        tau = np.minimum((self.phase_step + 1) / interp, 1.0)

        rows = np.arange(self.world_count)
        goal = self._phase_goal[index]
        goal_pos = self._goal_pos[goal, rows]
        goal_quat = self._goal_quat[goal, rows]

        self._command_pos = self._start_pos + (goal_pos - self._start_pos) * tau[:, None]
        self._command_quat = M.quat_slerp(self._start_quat, goal_quat, tau)

        wanted = self._phase_gripper_open[index].astype(float)
        self._grip_command += np.clip(wanted - self._grip_command, -self._grip_rate, self._grip_rate)
        return self._command_pos, self._command_quat, self._grip_command, interp

    def _solve_ik(self, group: _IKGroup, command_pos: np.ndarray, command_quat: np.ndarray) -> np.ndarray:
        """Warm-started IK for one arm entity. Returns ``(len(group.worlds), n_qs)``.

        ``max_samples=1`` is the whole reason this is a servo and not a planner: Genesis'
        default of 50 restarts *uniformly resamples every joint inside its limits* between
        samples, so a leg whose target drifts briefly out of reach comes back with an
        unrelated arm configuration and the next tick commands a violent reconfiguration
        through the board.  One sample makes the call a plain damped-least-squares descent
        from ``init_qpos``, which is what ``newton.ik`` does.
        """
        import torch

        worlds = group.worlds
        pos = np.ascontiguousarray(command_pos[worlds], dtype=np.float32)
        quat = np.ascontiguousarray(M.to_gs_quat(command_quat[worlds]), dtype=np.float32)
        # Genesis requires a tensor/ndarray with a leading env dimension; a list raises an
        # AttributeError from inside the shape check.
        solution, error = group.entity.inverse_kinematics(
            link=group.link,
            pos=torch.as_tensor(pos),
            quat=torch.as_tensor(quat),
            init_qpos=torch.as_tensor(group.q_warm),
            respect_joint_limit=True,
            max_samples=1,
            max_solver_iters=self.ik_iterations,
            damping=IK_DAMPING,
            max_step_size=IK_MAX_STEP,
            dofs_idx_local=group.dofs_local,
            return_error=True,
            envs_idx=group.env_rows,
        )
        solution = np.ascontiguousarray(M.as_numpy(solution).reshape(len(worlds), -1), dtype=np.float32)
        residual = M.as_numpy(error).reshape(len(worlds), 6)

        # Two failure modes, both silent in Genesis. The sentinel means the returned qpos was
        # never written and is uninitialised memory, so that world keeps its previous warm
        # start rather than teleporting; ordinary non-convergence is only counted, because a
        # target briefly outside the workspace is a normal event and tracking as close as
        # possible is the right response (and is what Newton's LM step does).
        garbage = np.abs(residual).max(axis=1) >= IK_ERROR_SENTINEL
        if garbage.any():
            solution[garbage] = group.q_warm[garbage]
        self._tick_ik_missed |= bool((np.linalg.norm(residual[:, :3], axis=1) > 1e-3).any())

        group.q_warm = solution
        return solution

    def _drive(self, command_pos: np.ndarray, command_quat: np.ndarray, grip: np.ndarray) -> None:
        """Solve IK for the commanded ``ee_link`` pose and write the joint targets."""
        # Latched across the groups and counted once, so the number stays comparable to
        # control_step_index however many arm entities the scene turned out to have.
        self._tick_ik_missed = False
        for group in self.ik_groups:
            solution = self._solve_ik(group, command_pos, command_quat)
            for row, world in enumerate(group.worlds):
                env = self._world_env[world]
                self._target_dofs[env, self._arm_dof_slots[world]] = solution[row, group.dofs_local]

        gripper = self._grip_close[None, :] + grip[:, None] * (self._grip_open - self._grip_close)[None, :]
        self._target_dofs[self._world_env[:, None], self._grip_dof_slots] = gripper
        self.scene.write_dof_targets(self._target_dofs)
        self.ik_failures += int(self._tick_ik_missed)

    def _advance(self, command_pos: np.ndarray, command_quat: np.ndarray, interp_steps: np.ndarray) -> None:
        """Roll a world over to its next phase once the leg is done *and* it arrived."""
        self.phase_step += 1
        active = self.phase_index < len(PHASES)
        index = np.minimum(self.phase_index, len(PHASES) - 1)

        hand_pos, hand_quat = self._hand_pose()
        pos_error = np.linalg.norm(hand_pos - command_pos, axis=-1)
        rot_error = M.quat_error_magnitude(hand_quat, command_quat)
        arrived = (pos_error < self._pos_tolerance[index]) & (rot_error < self._rot_tolerance[index])

        rollover = (
            active
            & (self.phase_step >= interp_steps)
            & (arrived | (self.phase_step >= self._phase_deadline[index]))
        )
        if not rollover.any():
            return

        holding = self._holding()
        for name in CAPTURE_AFTER:
            recompute = rollover & (self.phase_index == _PHASE_BY_NAME[name]) & holding
            for world in np.flatnonzero(recompute):
                self._capture_grasp(int(world), hand_pos, hand_quat)

        self._start_pos[rollover] = command_pos[rollover]
        self._start_quat[rollover] = command_quat[rollover]
        self.phase_index[rollover] += 1
        self.phase_step[rollover] = 0

    ##
    # Planning
    ##

    def _sample_command(self, world: int) -> None:
        """Draw the piece to move and where to put it, honouring ``--balance-kinds``."""
        movable = self._movable[world]
        kinds = self.scene.worlds[world].piece_kinds
        preferred = self._preferred_kind[world]
        if preferred is not None:
            filtered = [index for index in movable if kinds[index] == preferred]
            # A preference that no reachable piece satisfies falls back to the full list
            # rather than skipping the episode.
            movable = filtered or movable
            self._preferred_kind[world] = None

        self.piece_index[world] = int(self.rng.choice(movable))
        self.target_index[world] = int(self.rng.integers(len(self._targets[world])))
        self.target_pos_w[world] = self._targets[world][self.target_index[world]]
        self.planned_kind[world] = kinds[int(self.piece_index[world])]

    def _plan_grasp(self, world: int) -> None:
        """Score the GraspGen candidates against this board and seed the keypoints."""
        world_info = self.scene.worlds[world]
        index = int(self.piece_index[world])
        flat = int(self._piece_offset[world]) + index
        span = slice(int(self._piece_offset[world]), int(self._piece_offset[world]) + world_info.piece_count)

        piece_pose_w = M.pose_to_matrix(self._piece_pos[flat], self._piece_quat[flat])

        others = [i for i in range(span.start, span.stop) if i != flat]
        other_centers = self._piece_pos[others] if others else np.zeros((0, 3))
        other_kinds = [world_info.piece_kinds[i - span.start] for i in others]

        plan = self.planner.plan(
            world_info.piece_kinds[index],
            piece_pose_w,
            other_centers,
            other_kinds,
            float(self.lift_height[world]),
        )
        for name, (position, quat) in plan.goals.items():
            self._goal_pos[_GOAL_INDEX[name], world] = position
            self._goal_quat[_GOAL_INDEX[name], world] = quat
        self._plan_score[world] = plan.score
        self._plan_penetration[world] = plan.penetration

    def _capture_grasp(self, world: int, hand_pos: np.ndarray, hand_quat: np.ndarray) -> None:
        """Freeze the hand-to-piece transform and turn the target square into hand poses."""
        flat = int(self._piece_offset[world]) + int(self.piece_index[world])
        piece_pose_w = M.pose_to_matrix(self._piece_pos[flat], self._piece_quat[flat])
        hand_pose_w = M.pose_to_matrix(hand_pos[world], hand_quat[world])

        _, goals = G.place_goals(
            self.target_pos_w[world], piece_pose_w, hand_pose_w, float(self.lift_height[world])
        )
        for name, (position, quat) in goals.items():
            self._goal_pos[_GOAL_INDEX[name], world] = position
            self._goal_quat[_GOAL_INDEX[name], world] = quat

    ##
    # Observations and terminations
    ##

    def _refresh_state(self) -> None:
        """Pull the whole scene's state across in three kernel launches."""
        positions, quats = self.scene.read_link_poses()
        self._link_pos = positions
        self._link_quat = quats
        self._dofs = self.scene.read_dof_positions()
        self._piece_pos = positions[self._piece_env, self._piece_slot]
        self._piece_quat = quats[self._piece_env, self._piece_slot]

    def _hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World pose of every world's ``ee_link`` -- what the IK commands."""
        return (
            self._link_pos[self._world_env, self._ee_slot],
            self._link_quat[self._world_env, self._ee_slot],
        )

    def _tcp_pose(self) -> np.ndarray:
        pos, quat = self._hand_pose()
        return pos + M.quat_rotate(quat, np.asarray(self.spec.tcp_offset))

    def _commanded_flat(self) -> np.ndarray:
        return self._piece_offset + self.piece_index

    def _gripper_fraction(self) -> np.ndarray:
        """Mean opening fraction over the fingers, 0 = closed, 1 = open."""
        values = self._dofs[self._world_env[:, None], self._grip_dof_slots]
        fraction = (values - self._grip_close[None, :]) / self._grip_span[None, :]
        return fraction[:, self._grip_valid].mean(axis=1)

    def _holding(self) -> np.ndarray:
        """The shared ``piece_grasped`` predicate: pinched fingers near the piece.

        The state machine's "am I holding it" test and the success termination call this
        same function, deliberately, so they can never disagree.
        """
        fraction = self._gripper_fraction()
        pinched = (fraction > GRASP_MIN_FRACTION) & (fraction < GRASP_MAX_FRACTION)
        piece_pos = self._piece_pos[self._commanded_flat()]
        distance = np.linalg.norm(self._tcp_pose() - piece_pos, axis=-1)
        return pinched & (distance < GRASP_MAX_TCP_DISTANCE)

    def _piece_up_axis(self) -> np.ndarray:
        """World ``+Z`` of every piece, flat over worlds. Its z component is cos(tilt)."""
        return M.quat_to_matrix(self._piece_quat)[:, :, 2]

    def _evaluate_terminations(self) -> None:
        """Latch the first termination that fires per world.

        A world is judged the moment its schedule runs out, not at the step budget.
        ``settle`` is the last leg, so a world at ``phase=done`` has already done everything
        it was going to do: running it on to the deadline changes nothing except how long the
        rest of the batch has to wait for it (all worlds reset together).  That also keeps
        ``timed_out`` meaning what it says -- the budget ran out *mid*-schedule -- instead of
        covering every failure that reached the end of the schedule.
        """
        commanded = self._commanded_flat()
        piece_pos = self._piece_pos
        up_z = self._piece_up_axis()[:, 2]

        fallen = np.zeros(self.world_count, dtype=bool)
        np.logical_or.at(
            fallen, self._piece_world, piece_pos[:, 2] - self._piece_origin_z < MIN_PIECE_HEIGHT
        )

        tilted = up_z < math.cos(math.radians(DISTURB_MAX_TILT_DEG))
        tilted[commanded] = False  # the piece being moved is allowed to be tipped
        disturbed = np.zeros(self.world_count, dtype=bool)
        np.logical_or.at(disturbed, self._piece_world, tilted)

        velocities = self.scene.read_link_velocities()
        speed = np.linalg.norm(velocities[self._piece_env, self._piece_slot], axis=-1)
        offset = piece_pos[commanded] - self.target_pos_w
        # The success predicate, kept clause by clause rather than reduced in place: a world
        # that finished its schedule without succeeding is labelled with the first clause
        # that rejected it, in this order, which is the whole difference between
        # "missed_target, 383 mm out" and "timed_out".
        clauses = {
            "missed_target": np.linalg.norm(offset[:, :2], axis=-1) < PLACE_XY_TOLERANCE,
            "off_surface": np.abs(offset[:, 2]) < PLACE_Z_TOLERANCE,
            "piece_tipped": up_z[commanded] > math.cos(math.radians(PLACE_MAX_TILT_DEG)),
            "not_released": ~self._holding(),
            "still_moving": speed[commanded] < PLACE_MAX_SPEED,
        }
        success = np.logical_and.reduce(list(clauses.values()))

        schedule_done = self.phase_index >= len(PHASES)
        timed_out = self.control_step_index >= self.max_steps
        for world in np.flatnonzero(~self.done):
            if success[world]:
                outcome = "success"
            elif fallen[world]:
                outcome = "piece_off_board"
            elif disturbed[world]:
                outcome = "board_disturbed"
            elif schedule_done[world]:
                outcome = next(label for label, holds in clauses.items() if not holds[world])
            elif timed_out:
                outcome = "timed_out"
            else:
                continue
            self.done[world] = True
            self.outcome[world] = outcome
            self._outcome_step[world] = self.control_step_index
            self._outcome_phase[world] = self._phase_name(int(world))
            # Captured here rather than at collection time: the other worlds keep running
            # until the slowest one finishes, and a dropped piece rolls.
            self._outcome_place_error[world] = np.linalg.norm(offset[world, :2])

    def _assign_preferred_kinds(self) -> None:
        """Round-robin the next episode's worlds onto the least-attempted kinds.

        Steers on *attempts*, not successes: ranking by successes fixates, because a kind the
        arm cannot pick stays the scarcest forever and absorbs every episode.
        """
        ranked = sorted(self._all_kinds, key=lambda kind: self.attempted_kinds[kind])
        for world in range(self.world_count):
            self._preferred_kind[world] = ranked[world] if world < len(ranked) else None

    ##
    # Reporting
    ##

    def _phase_name(self, world: int) -> str:
        index = int(self.phase_index[world])
        return PHASES[index].name if index < len(PHASES) else "done"

    def report(self, world: int = 0) -> str:
        """One-line snapshot of a world, for ``--debug``."""
        world_info = self.scene.worlds[world]
        index = int(self.piece_index[world])
        flat = int(self._piece_offset[world]) + index
        hand_pos, hand_quat = self._hand_pose()
        piece_pos = self._piece_pos[flat]
        tracking = float(np.linalg.norm(hand_pos[world] - self._command_pos[world]))
        rotation = float(M.quat_error_magnitude(hand_quat[world], self._command_quat[world]))
        return (
            f"[world{world}] step={self.control_step_index:3d} {self._phase_name(world):<10s}"
            f" {world_info.piece_kinds[index]:<6s}"
            f" track={tracking * 1000:6.1f}mm/{math.degrees(rotation):5.1f}deg"
            f" frac={self._gripper_fraction()[world]:.2f} hold={int(self._holding()[world])}"
            f" piece_z={(piece_pos[2] - world_info.table_top_z - world_info.origin[2]) * 1000:6.1f}mm"
            f" to_target={float(np.linalg.norm(piece_pos[:2] - self.target_pos_w[world, :2])) * 1000:6.1f}mm"
            f" {self.outcome[world]}"
        )

    def describe(self) -> str:
        """Static summary of what the task is about to run."""
        lines = [
            f"  arm={self.spec.key} ee_link={self.spec.ee_link} stroke={self.spec.max_opening * 1000:.0f}mm"
            f" grasps={self.library.path.name} candidates={self.library.max_candidates}"
            f" yaws={self.planner.num_yaws}",
            f"  control={1.0 / self.control_dt:.0f}Hz frames/tick={self.frames_per_tick}"
            f" ik_iters={self.ik_iterations} ik_groups={len(self.ik_groups)}"
            f" episode<={self.max_steps} steps ({self.max_steps * self.control_dt:.0f}s)",
        ]
        for world in self.scene.worlds:
            index = world.index
            lines.append(
                f"  world {index}: {world.layout.scenario:<6s} pieces={world.piece_count:2d}"
                f" movable={len(self._movable[index]):2d} targets={len(self._targets[index]):2d}"
                f" carry={self.lift_height[index] * 1000:.0f}mm"
            )
        return "\n".join(lines)
