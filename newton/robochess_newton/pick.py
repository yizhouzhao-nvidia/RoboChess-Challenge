"""Scripted pick-and-place of a chess piece, batched over the worlds of a scene.

The Newton counterpart of ``ChessPickPolicy`` in
``lab/scripts/generate_chess_pick_demos.py``.  Each episode asks one question --
move piece *i* to destination *j* -- and answers it with a nine-phase schedule
(``pre_grasp, descend, close, lift, transfer, place, release, retreat, settle``)
whose keypoints come from the GraspGen grasp chosen by
:class:`~robochess_newton.grasps.GraspPlanner`.

Three things about the schedule are not cosmetic and are reproduced exactly:

* **Each phase is a time interpolation *and* an arrival test.**  Inverse kinematics
  takes a bounded step per control tick, so a purely time-triggered schedule closes
  the fingers wherever the arm happens to be -- tens of millimetres short of the
  grasp, which slides the piece out.  A phase therefore holds at its goal until the
  pose error is inside its own tolerance, with a per-phase deadline so a bad IK
  solution cannot stall the episode.
* **The next leg starts from the previous *command*, not the measured pose.**  That
  keeps the commanded trajectory continuous instead of baking each leg's tracking
  error into the start of the next one.
* **The place pose is derived from the measured hand-to-piece transform**, refreshed
  at the end of ``close``, ``lift`` and ``transfer`` and only while the piece is
  actually held.  Deriving it from a failed grasp aims the arm at a pose metres
  away and turns a quiet failure into a thrashing one.

Two details are deliberately *not* reproduced, both because ``newton.ik`` tracks the
command far more tightly than the differential-IK action term the Isaac Lab schedule
was tuned against, which turns two of its latent quirks into real failures:

* the leg's last commanded pose is the goal, not ``(N-1)/N`` of the way to it
  (:meth:`ChessPickTask._compute`);
* ``transfer`` ends at the carry height rather than a fixed 120 mm, so the carry is
  level instead of a descending sweep across the pieces still standing
  (:data:`~robochess_newton.grasps.PLACE_APPROACH_HEIGHT`).

Measured, franka, 24 episodes each at seed 0: ``1d`` 71 % -> 83 % with the level
carry -> 88 % with both, ``4x4`` 100 % throughout.

Isaac Lab drove this with a differential-IK *action term* in the robot root frame.
Newton has ``newton.ik``, a batched Levenberg-Marquardt solver over an arbitrary set
of objectives, and its targets are in **world** coordinates -- so the whole state
machine here works in world coordinates.  The two are equivalent: every robot root
is axis-aligned with the world, so root and world differ by a pure translation and
both the position tolerance and the rotation tolerance are unchanged.

The IK model is the arm *alone* (:meth:`ChessScene.robot_only_builder`).  Solving
over the scene model would let the solver "reach" a target by teleporting a chess
piece, since the pieces' free joints are DOFs too, and ``joint_dof_mask`` only
exists on newton 1.6.

**Batching.** The IK solve and the physics are batched over worlds on the GPU; the
schedule itself is numpy vectorised over worlds on the host, one array op per
quantity rather than a Python loop, and runs at 30 Hz.  The only per-world Python
is grasp planning, which happens once per episode.  This is a deliberate departure
from ``newton/examples/ik/example_ik_cube_stacking.py``, which puts the equivalent
logic in two warp kernels: the per-phase tolerance table, the deadline logic and
the grasp re-scoring are all data-dependent branching that reads far more clearly
in numpy, and at 30 Hz over <= 16 worlds the host cost is ~0.3 ms per tick against
a ~25 ms physics step.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# openusd's UsdPhysics parser segfaults nondeterministically under threads; every
# asset in the scene goes through it. Before any import that can reach pxr.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

# grasps first: it imports board_layout, which strips the repo root from sys.path,
# where this repo's own newton/ directory would shadow the real package.
from . import grasps as G
from .robots import NewtonRobotSpec, get_spec, joint_target_positions
from .scene import ChessScene

import numpy as np
import warp as wp

import newton
import newton.ik as ik

__all__ = [
    "ARM_JOINT_JITTER",
    "CONTROL_DT",
    "DISTURB_MAX_TILT_DEG",
    "EPISODE_LENGTH_S",
    "GRASP_MAX_FRACTION",
    "GRASP_MAX_TCP_DISTANCE",
    "GRASP_MIN_FRACTION",
    "GRIPPER_TRAVEL_TIME",
    "IK_ITERATIONS",
    "IK_LAMBDA_INITIAL",
    "MIN_PIECE_HEIGHT",
    "PHASES",
    "PIECE_XY_JITTER",
    "PIECE_YAW_JITTER",
    "PLACE_MAX_SPEED",
    "PLACE_MAX_TILT_DEG",
    "PLACE_XY_TOLERANCE",
    "PLACE_Z_TOLERANCE",
    "PICK_ARM_ARMATURE",
    "PICK_ARM_KE",
    "PICK_DAMPING_RATIO",
    "PICK_GRIPPER_ARMATURE",
    "PICK_GRIPPER_KE",
    "ChessPickTask",
    "EpisodeResult",
    "Phase",
    "apply_pick_gains",
]

##
# Rates. The Isaac Lab task runs sim at 1/120 with decimation 4; the Newton scene
# was validated at 1/240, so a control tick is two 60 Hz frames instead of one.
##

CONTROL_DT = 1.0 / 30.0
"""Control period [s]. One IK solve and one schedule tick per period."""

EPISODE_LENGTH_S = 24.0
"""Episode budget [s]. The nominal schedule is 10.5 s and its worst case 20.5 s.

Every phase rolls over at its own deadline and a world is judged as soon as the
schedule ends, so at this budget nothing reaches it: ``timed_out`` is the guard
against a shortened ``episode_length_s``, not the usual failure label."""

IK_ITERATIONS = 24
"""Levenberg-Marquardt iterations per control tick, warm-started from the previous solve."""

IK_LAMBDA_INITIAL = 0.1
"""Initial LM damping; the value ``example_ik_cube_stacking.py`` reaches 1 mm with."""

GRIPPER_TRAVEL_TIME = 0.5
"""How long the fingers take to cross their full stroke [s].

Isaac Lab's ``BinaryJointPositionAction`` steps the finger targets from open to
closed in one tick, and its PhysX implicit actuators absorb that.  A MuJoCo
position actuator does not: the step is a 40 mm command error against the
picking stiffness, i.e. hundreds of newtons on the first substep, and a 25 g pawn
is flicked out of the hand before the jaws touch it.  Measured on franka/3x3 with
the hard step: the pawn is knocked flat during ``close`` in every episode
(``piece_z`` jumps to 25 mm, its own radius, and the gripper fraction reaches 0.00
with nothing in it).  Ramping the command over 0.5 s -- shorter than the 0.9 s
``close`` phase, so the schedule is unchanged -- keeps the jaws below the piece's
tipping force.  ``example_ik_cube_stacking.py`` ramps its gripper for the same
reason."""

##
# Drive tuning for picking. robots.py ships 2000/100 on every DOF and no armature,
# which holds any of these arms to 1e-10 rad of a *static* pose once gravity is
# compensated -- enough to visualise a board. Closing a gripper on a 25 g piece is a
# different load case, and the numbers below come from franka/3x3 and franka/1d
# sweeps rather than guesses -- except PICK_GRIPPER_ARMATURE, whose docstring says
# what re-measuring it actually showed.
##

PICK_DAMPING_RATIO = 1.0 / 20.0
"""Position-drive damping as a fraction of the stiffness, on the arm and the gripper.

``load_robot`` ships 2000/100, the same ratio, and every stiffness tried while tuning
the pick was damped this way, so :func:`apply_pick_gains` scales the damping with
whatever stiffness it is handed rather than leaving a stiffer drive underdamped."""

PICK_ARM_KE = 4000.0
"""Position-drive stiffness on the arm DOFs.

The open Franka hand is 106 mm across the fingers' outer faces and a board square is
84 mm, so descending onto a piece grazes its neighbours.  At 2000/100 that graze
stops the hand 23 mm short of the grasp for the whole descend and the fingers close
on air (measured, franka/1d, newton 1.6).  4000 is also where Isaac Lab's own
config sits (4500 on the Franka's forearm, 4000 on piper/rebot).  Going further --
8000, 12000, 20000 were all tried -- buys nothing and starts costing success,
because a stiffer arm pushes harder into the neighbour it is grazing."""

PICK_GRIPPER_KE = 600.0
"""Position-drive stiffness on the gripper DOFs -- deliberately *not* the arm's.

A MuJoCo position actuator applies ``ke`` times the position error and the close
command is the jaws' hard stop, so once a jaw touches the piece the command keeps
running past it and the squeeze grows without limit.  At 2000 that is tens of
newtons on a 25 g pawn gripped at a 11 mm waist, and the piece is squeezed out of
the tapered grip like a bar of soap (measured on franka/3x3: the piece is ejected
upward at 0.45 m/s during the lift).  600 settles at ~3 N, which is a firm grip on
the heaviest piece here (a 103 g king) and does not wedge."""

PICK_ARM_ARMATURE = 0.3
"""Rotor inertia added to every arm DOF [kg m^2].

**This is the single change that makes the grasp hold at all**, and it is easy to
miss because nothing about a static pose reveals it.  Newton's URDF/MJCF importers
leave ``joint_armature`` at 0 for all four of these arms, so a 15 g finger driven by
a position actuator against a contact is a numerically stiff system that MuJoCo's
15-iteration solve cannot hold: the piece either shoots out of the jaws or slides
through them and topples.  Re-measured on franka/3x3, 4 worlds x 8 episodes at
seed 0, everything else at the shipped defaults:

    --arm-armature 0.00 -> 0/8    0.05 -> 8/8    0.10 -> 8/8    0.30 -> 8/8

At 0.00 all eight run the schedule out and end ``missed_target`` 45-365 mm from the
destination, i.e. the piece is never carried anywhere.  So it is the *presence* of
the armature that decides the pick, not its size anywhere above ~0.05; 0.3 is kept
because ``newton/examples/ik/example_ik_cube_stacking.py`` sets exactly these values
(0.3 on the shoulder joints, 0.11 on the wrist, 0.15 on the fingers) and reaches
1 mm tracking with them, so they were tried first here."""

PICK_GRIPPER_ARMATURE = 0.15
"""Rotor inertia added to every gripper DOF [kg m^2].

Unlike :data:`PICK_ARM_ARMATURE`, **no case tried here is sensitive to this value**:
franka/3x3 is 8/8 at 0.00, at 0.15 and at 0.30 (4 worlds x 8 episodes, seed 0) and
yam/4x4 is 12/16 at 0.00 and at the default (8 worlds x 16 episodes, seed 0).  It
does reach the solver -- ``model.joint_armature`` on the finger DOFs reads 0.15
after :func:`apply_pick_gains` -- so it is live rather than dead code.  It is
retained as the counterpart of the arm value and because
``example_ik_cube_stacking.py`` arms its fingers the same way, not because dropping
it was ever traced to a failure."""


def apply_pick_gains(
    scene: ChessScene,
    arm_ke: float = PICK_ARM_KE,
    gripper_ke: float = PICK_GRIPPER_KE,
    arm_armature: float = PICK_ARM_ARMATURE,
    gripper_armature: float = PICK_GRIPPER_ARMATURE,
) -> None:
    """Retune every arm in *scene* for picking. Must run **before** ``finalize()``.

    ``load_robot`` applies one stiffness to every DOF of an arm and no armature at
    all, which is the right default for holding a pose and the wrong one for closing
    a gripper.  The builder is the only place these can be split without a second set
    of ``NewtonRobotSpec`` fields, and it is still open until the caller finalizes.
    """
    if scene.model is not None:
        raise RuntimeError("apply_pick_gains() must be called before ChessScene.finalize()")
    builder = scene.builder
    for world in scene.worlds:
        for dof in world.robot.arm_dofs:
            builder.joint_target_ke[dof] = arm_ke
            builder.joint_target_kd[dof] = arm_ke * PICK_DAMPING_RATIO
            builder.joint_armature[dof] = arm_armature
        for dof in world.robot.gripper_dofs:
            builder.joint_target_ke[dof] = gripper_ke
            builder.joint_target_kd[dof] = gripper_ke * PICK_DAMPING_RATIO
            builder.joint_armature[dof] = gripper_armature


##
# Reset randomisation, from mdp/events.py.
##

PIECE_XY_JITTER = 0.004
"""Uniform +-XY offset applied to every piece at reset [m]."""

PIECE_YAW_JITTER = 0.35
"""Uniform +-yaw applied to every piece at reset [rad]."""

ARM_JOINT_JITTER = 0.02
"""Uniform +-offset applied to the arm joints at reset [rad].

Applied to the arm DOFs only.  Isaac Lab's ``reset_joints_by_offset`` also hits the
finger joints, which moves the gripper's open fraction by up to half its range and
makes the ``piece_grasped`` predicate fire on an empty hand at reset; that is a
quirk of the lab config, not part of the task."""

##
# Success / termination, from mdp/terminations.py and mdp/observations.py.
##

GRASP_MAX_TCP_DISTANCE = 0.18
"""How far the TCP may be from the piece origin and still count as holding it [m].

Wide enough to cover the knight, which GraspGen grips off-axis by the head."""

GRASP_MIN_FRACTION = 0.04
GRASP_MAX_FRACTION = 0.94
"""Gripper-opening fraction bracket that counts as pinched.

Averaged over the fingers, never tested per finger: the fingers are independently
actuated and close asymmetrically on an off-centre grasp."""

PLACE_XY_TOLERANCE = 0.02
PLACE_Z_TOLERANCE = 0.01
PLACE_MAX_SPEED = 0.05
PLACE_MAX_TILT_DEG = 25.0
DISTURB_MAX_TILT_DEG = 30.0
"""Tilt beyond which a piece *other* than the commanded one counts as knocked over."""

MIN_PIECE_HEIGHT = 0.5
"""A piece below this world height has left the table [m] (the table top is at 0.77)."""


@dataclass(frozen=True)
class Phase:
    """One leg of the pick-and-place: interpolate to ``goal``, then wait for arrival."""

    name: str
    duration: float
    """Interpolation time [s]; also the minimum time the leg takes."""

    gripper_open: bool
    goal: str
    """Which entry of :data:`~robochess_newton.grasps.GOAL_NAMES` this leg drives to."""

    pos_tolerance: float = 0.004
    rot_tolerance: float = 0.06
    settle_timeout: float = 1.5
    """Extra time [s] the leg may spend waiting to arrive before it gives up and
    rolls over anyway."""


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

Three points rather than one: the fingers may still be travelling when ``close``
ends, and the piece can settle or slip during the lift and the carry."""

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

    ``piece_off_board`` and ``board_disturbed`` are the two board-wrecking
    terminations; ``timed_out`` means the step budget ran out with the schedule still
    running.  A schedule that ran to the end without placing is labelled with the
    first clause of the success predicate that rejected it -- ``missed_target``,
    ``off_surface``, ``piece_tipped``, ``not_released`` or ``still_moving`` -- rather
    than being lumped in with the timeouts."""

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


class ChessPickTask:
    """Runs the nine-phase pick schedule in every world of a :class:`ChessScene`.

    The scene must be finalized and every world must carry the *same* arm (the IK
    solver shares one robot model across problems); scenarios may differ.
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
        if scene.model is None:
            raise RuntimeError("ChessScene.finalize() must be called before ChessPickTask")
        specs = {world.spec.key for world in scene.worlds}
        if len(specs) != 1:
            raise ValueError(f"every world must use the same arm; got {sorted(specs)}")

        self.scene = scene
        self.spec: NewtonRobotSpec = get_spec(next(iter(specs)))
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
        """Cache every model index the tick loop needs, as numpy arrays."""
        scene, worlds = self.scene, self.scene.worlds
        spec = self.spec

        self._ee_bodies = np.array([world.robot.ee_body for world in worlds], dtype=np.int64)
        self._arm_dofs = np.array([world.robot.arm_dofs for world in worlds], dtype=np.int64)
        self._grip_dofs = np.array([world.robot.gripper_dofs for world in worlds], dtype=np.int64)
        self._arm_coords = np.array([world.arm_coords for world in worlds], dtype=np.int64)
        self._grip_coords = np.array([world.gripper_coords for world in worlds], dtype=np.int64)
        self._origins = np.array([world.origin for world in worlds], dtype=np.float64)

        self._grip_open = np.asarray(spec.gripper_open, dtype=np.float64)
        self._grip_close = np.asarray(spec.gripper_close, dtype=np.float64)
        span = self._grip_open - self._grip_close
        self._grip_valid = np.abs(span) > 1e-9
        if not self._grip_valid.any():
            raise ValueError(f"{spec.key}: open and close commands are identical, cannot detect a grasp")
        self._grip_span = np.where(self._grip_valid, span, 1.0)

        # Pieces are ragged across worlds (a 1d board has 6, an 8x8 has 32), so they
        # live in one flat array plus a world id, which keeps every per-piece test a
        # single vectorised op.
        bodies, coords, world_of, offsets = [], [], [], []
        for index, world in enumerate(worlds):
            offsets.append(len(bodies))
            bodies.extend(world.piece_bodies)
            coords.extend(world.piece_coords)
            world_of.extend([index] * world.piece_count)
        self._piece_bodies = np.array(bodies, dtype=np.int64)
        self._piece_coords = np.array(coords, dtype=np.int64)
        self._piece_world = np.array(world_of, dtype=np.int64)
        self._piece_offset = np.array(offsets, dtype=np.int64)

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
            [G.carry_height(max(self.piece_geometry[k]["height"] for k in world.piece_kinds), spec.reach)
             for world in worlds]
        )

        # joint_q holds the pristine spawn state: the arm at its home pose and every
        # piece on its square. Reset perturbs a copy of this.
        self._default_joint_q = scene.model.joint_q.numpy().copy()
        # Seeded from the control array once: every DOF this task does not write is a
        # piece's free joint, whose target mode is NONE, so the rest never changes and
        # the array does not have to be read back each tick.
        self._target_np = joint_target_positions(scene.control).numpy().copy()

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
        self._body_q = np.zeros((self.scene.model.body_count, 7))
        self._joint_q = np.zeros(self.scene.model.joint_coord_count)

    def _setup_ik(self) -> None:
        """One robot-only model, one LM solver, ``world_count`` problems."""
        scene = self.scene
        builder, handle = scene.robot_only_builder(0)
        self.ik_model = builder.finalize(device=scene.model.device)
        self.ik_dofs = self.ik_model.joint_coord_count
        self._arm_ik_slots = np.asarray(self.spec.arm_dofs, dtype=np.int64)

        worlds = self.world_count
        self._ik_target_pos = wp.zeros(worlds, dtype=wp.vec3, device=scene.model.device)
        self._ik_target_rot = wp.zeros(worlds, dtype=wp.vec4, device=scene.model.device)
        self._pos_objective = ik.IKObjectivePosition(
            link_index=handle.ee_body,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=self._ik_target_pos,
        )
        self._rot_objective = ik.IKObjectiveRotation(
            link_index=handle.ee_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self._ik_target_rot,
        )
        lower = np.tile(self.ik_model.joint_limit_lower.numpy(), worlds)
        upper = np.tile(self.ik_model.joint_limit_upper.numpy(), worlds)
        self._limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(lower, dtype=wp.float32, device=scene.model.device),
            joint_limit_upper=wp.array(upper, dtype=wp.float32, device=scene.model.device),
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=worlds,
            objectives=[self._pos_objective, self._rot_objective, self._limit_objective],
            lambda_initial=IK_LAMBDA_INITIAL,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.joint_q_ik = wp.zeros((worlds, self.ik_dofs), dtype=wp.float32, device=scene.model.device)
        self._joint_q_ik_np = np.zeros((worlds, self.ik_dofs), dtype=np.float32)

    ##
    # Episode lifecycle
    ##

    def reset_episode(self) -> None:
        """Randomise the scene, sample a move per world and plan its grasp.

        Every world resets together.  Isaac Lab auto-resets each environment the
        moment it terminates; here the caller reads the outcomes first, which makes
        the per-episode accounting exact and costs only the worlds that finish early
        idling until the slowest one is done.
        """
        scene = self.scene
        joint_q = self._default_joint_q.copy()

        arm_home = joint_q[self._arm_coords] + self.rng.uniform(
            -ARM_JOINT_JITTER, ARM_JOINT_JITTER, size=self._arm_coords.shape
        )
        joint_q[self._arm_coords] = arm_home
        joint_q[self._grip_coords] = self._grip_open

        # Free joints store 7 coordinates: xyz then the xyzw quaternion.
        coords = self._piece_coords
        count = len(coords)
        offsets = self.rng.uniform(-PIECE_XY_JITTER, PIECE_XY_JITTER, size=(count, 2))
        joint_q[coords] += offsets[:, 0]
        joint_q[coords + 1] += offsets[:, 1]
        yaw = self.rng.uniform(-PIECE_YAW_JITTER, PIECE_YAW_JITTER, size=count)
        spin = np.stack([np.zeros(count), np.zeros(count), np.sin(yaw / 2.0), np.cos(yaw / 2.0)], axis=-1)
        base = np.stack([joint_q[coords + 3 + i] for i in range(4)], axis=-1)
        spun = G.quat_mul(base, spin)
        for i in range(4):
            joint_q[coords + 3 + i] = spun[:, i]

        scene.state_0.joint_q.assign(joint_q.astype(np.float32))
        scene.state_0.joint_qd.zero_()
        scene.state_1.joint_qd.zero_()
        newton.eval_fk(scene.model, scene.state_0.joint_q, scene.state_0.joint_qd, scene.state_0)
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

        # Warm-start the IK from the arm's actual (jittered) posture, so the first
        # command -- the measured hand pose -- is already a solution.
        for world in range(self.world_count):
            start = self.scene.worlds[world].coord_start
            self._joint_q_ik_np[world] = joint_q[start : start + self.ik_dofs]
        self.joint_q_ik.assign(self._joint_q_ik_np)

        for world in range(self.world_count):
            self._sample_command(world)
            self._plan_grasp(world)

    @property
    def episode_finished(self) -> bool:
        return bool(self.done.all()) or self.control_step_index >= self.max_steps

    def collect_results(self) -> list[EpisodeResult]:
        """Score the episode, one result per *finished* world; call once ``episode_finished``.

        A world that ran its episode out always has an outcome latched by
        :meth:`_evaluate_terminations` -- a termination, the placement verdict at the
        end of the schedule, or ``timed_out`` at the step budget.  So an unlatched
        world can only mean the caller stopped stepping early (a frame budget), and
        those are **not** returned: scoring a world still in ``descend`` would count
        it as a failed attempt and pull the reported success rate down with an
        episode that never got the chance to finish.
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

        *on_frame* is called after every rendered 60 Hz frame, which is where a
        viewer hooks in.
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

        The one place this deliberately departs from ``ChessPickPolicy``.  Isaac Lab
        computes ``tau = phase_step / N`` *before* incrementing ``phase_step``, so the
        first rollover opportunity comes while ``tau = (N-1)/N`` and the exact goal is
        only ever commanded by a leg that failed to arrive on time.  Under a
        differential-IK action term that is invisible, because the arm never arrives
        on the first opportunity.  ``newton.ik`` tracks to 0.2 mm, so here it fires
        every time and every leg stops one step short -- 3 mm on ``descend``, which is
        half the bishop's 6.9 mm neck.  Counting from ``phase_step + 1`` makes the
        last commanded pose the goal itself at no cost in schedule length.
        """
        index = np.minimum(self.phase_index, len(PHASES) - 1)
        interp = self._phase_steps[index]
        tau = np.minimum((self.phase_step + 1) / interp, 1.0)

        rows = np.arange(self.world_count)
        goal = self._phase_goal[index]
        goal_pos = self._goal_pos[goal, rows]
        goal_quat = self._goal_quat[goal, rows]

        self._command_pos = self._start_pos + (goal_pos - self._start_pos) * tau[:, None]
        self._command_quat = G.quat_slerp(self._start_quat, goal_quat, tau)

        wanted = self._phase_gripper_open[index].astype(float)
        self._grip_command += np.clip(wanted - self._grip_command, -self._grip_rate, self._grip_rate)
        return self._command_pos, self._command_quat, self._grip_command, interp

    def _drive(self, command_pos: np.ndarray, command_quat: np.ndarray, grip: np.ndarray) -> None:
        """Solve IK for the commanded ``ee_body`` pose and write the joint targets."""
        # newton.ik targets are world positions in the *IK model's* frame, and the IK
        # model holds one arm at its base pose with no world offset, so replicated
        # worlds have to be folded back onto world 0.
        self._ik_target_pos.assign(np.ascontiguousarray(command_pos - self._origins, dtype=np.float32))
        self._ik_target_rot.assign(np.ascontiguousarray(command_quat, dtype=np.float32))
        self._pos_objective.set_target_positions(self._ik_target_pos)
        self._rot_objective.set_target_rotations(self._ik_target_rot)
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iterations)

        solution = self.joint_q_ik.numpy()
        self._target_np[self._arm_dofs] = solution[:, self._arm_ik_slots]
        self._target_np[self._grip_dofs] = (
            self._grip_close[None, :] + grip[:, None] * (self._grip_open - self._grip_close)[None, :]
        )
        joint_target_positions(self.scene.control).assign(self._target_np)

    def _advance(self, command_pos: np.ndarray, command_quat: np.ndarray, interp_steps: np.ndarray) -> None:
        """Roll a world over to its next phase once the leg is done *and* it arrived."""
        self.phase_step += 1
        active = self.phase_index < len(PHASES)
        index = np.minimum(self.phase_index, len(PHASES) - 1)

        hand_pos, hand_quat = self._hand_pose()
        pos_error = np.linalg.norm(hand_pos - command_pos, axis=-1)
        rot_error = G.quat_error_magnitude(hand_quat, command_quat)
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
            # A preference that no reachable piece satisfies falls back to the full
            # list rather than skipping the episode.
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

        pose = self._body_q[self._piece_bodies[flat]]
        piece_pose_w = G.pose_to_matrix(pose[:3], pose[3:7])

        others = [i for i in range(span.start, span.stop) if i != flat]
        other_centers = self._body_q[self._piece_bodies[others], :3] if others else np.zeros((0, 3))
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
        pose = self._body_q[self._piece_bodies[flat]]
        piece_pose_w = G.pose_to_matrix(pose[:3], pose[3:7])
        hand_pose_w = G.pose_to_matrix(hand_pos[world], hand_quat[world])

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
        self._body_q = self.scene.state_0.body_q.numpy().astype(np.float64)
        self._joint_q = self.scene.state_0.joint_q.numpy().astype(np.float64)

    def _hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World pose of every world's ``ee_body`` -- what the IK commands."""
        pose = self._body_q[self._ee_bodies]
        return pose[:, :3], pose[:, 3:7]

    def _tcp_pose(self) -> np.ndarray:
        pose = self._body_q[self._ee_bodies]
        return pose[:, :3] + G.quat_rotate(pose[:, 3:7], np.asarray(self.spec.tcp_offset))

    def _commanded_flat(self) -> np.ndarray:
        return self._piece_offset + self.piece_index

    def _gripper_fraction(self) -> np.ndarray:
        """Mean opening fraction over the fingers, 0 = closed, 1 = open."""
        values = self._joint_q[self._grip_coords]
        fraction = (values - self._grip_close[None, :]) / self._grip_span[None, :]
        return fraction[:, self._grip_valid].mean(axis=1)

    def _holding(self) -> np.ndarray:
        """The shared ``piece_grasped`` predicate: pinched fingers near the piece.

        The state machine's "am I holding it" test and the success termination call
        this same function, deliberately, so they can never disagree.
        """
        fraction = self._gripper_fraction()
        pinched = (fraction > GRASP_MIN_FRACTION) & (fraction < GRASP_MAX_FRACTION)
        piece_pos = self._body_q[self._piece_bodies[self._commanded_flat()], :3]
        distance = np.linalg.norm(self._tcp_pose() - piece_pos, axis=-1)
        return pinched & (distance < GRASP_MAX_TCP_DISTANCE)

    def _piece_up_axis(self) -> np.ndarray:
        """World ``+Z`` of every piece, flat over worlds. Its z component is cos(tilt)."""
        return G.quat_to_matrix(self._body_q[self._piece_bodies, 3:7])[:, :, 2]

    def _evaluate_terminations(self) -> None:
        """Latch the first termination that fires per world.

        A world is judged the moment its schedule runs out, not at the step budget.
        ``settle`` is the last leg, so a world at ``phase=done`` has already done
        everything it was going to do: running it on to the deadline changes nothing
        except how long the rest of the batch has to wait for it (all worlds reset
        together).  That also keeps ``timed_out`` meaning what it says -- the budget
        ran out *mid*-schedule -- instead of covering every failure that reached the
        end of the schedule.
        """
        commanded = self._commanded_flat()
        piece_pos = self._body_q[self._piece_bodies, :3]
        up_z = self._piece_up_axis()[:, 2]

        fallen = np.zeros(self.world_count, dtype=bool)
        height = piece_pos[:, 2] - self._origins[self._piece_world, 2]
        np.logical_or.at(fallen, self._piece_world, height < MIN_PIECE_HEIGHT)

        tilted = up_z < math.cos(math.radians(DISTURB_MAX_TILT_DEG))
        tilted[commanded] = False  # the piece being moved is allowed to be tipped
        disturbed = np.zeros(self.world_count, dtype=bool)
        np.logical_or.at(disturbed, self._piece_world, tilted)

        speed = np.linalg.norm(self.scene.state_0.body_qd.numpy()[self._piece_bodies, :3], axis=-1)
        offset = piece_pos[commanded] - self.target_pos_w
        # The success predicate, kept clause by clause rather than reduced in place: a
        # world that finished its schedule without succeeding is labelled with the
        # first clause that rejected it, in this order, which is the whole difference
        # between "missed_target, 383 mm out" and "timed_out".
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
            # Captured here rather than at collection time: the other worlds keep
            # running until the slowest one finishes, and a dropped piece rolls.
            self._outcome_place_error[world] = np.linalg.norm(offset[world, :2])

    def _assign_preferred_kinds(self) -> None:
        """Round-robin the next episode's worlds onto the least-attempted kinds.

        Steers on *attempts*, not successes: ranking by successes fixates, because a
        kind the arm cannot pick stays the scarcest forever and absorbs every
        episode.
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
        piece_pos = self._body_q[self._piece_bodies[flat], :3]
        tracking = float(np.linalg.norm(hand_pos[world] - self._command_pos[world]))
        rotation = float(G.quat_error_magnitude(hand_quat[world], self._command_quat[world]))
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
            f"  arm={self.spec.key} ee_body={self.spec.ee_body} stroke={self.spec.max_opening * 1000:.0f}mm"
            f" grasps={self.library.path.name} candidates={self.library.max_candidates}"
            f" yaws={self.planner.num_yaws}",
            f"  control={1.0 / self.control_dt:.0f}Hz frames/tick={self.frames_per_tick}"
            f" ik_iters={self.ik_iterations} ik_dofs={self.ik_dofs}"
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
