"""The RoboChess table scene, assembled into a :class:`newton.ModelBuilder`.

This is the Newton counterpart of ``FrankaChessSceneCfg`` +
``configure_chess_scene`` + ``ChessPickEnvCfg.set_robot`` on the Isaac Lab side:
ground plane, table, chess board, one free rigid body per piece, the capture tray
and one arm, with the board placed ``spec.board_distance`` in front of that arm
exactly as ``ChessPickEnvCfg.board_center()`` does.

Isaac Lab hands a task a named scene graph and resolves ``"piece_white_pawn_0"``
at runtime.  Newton has no such registry -- a finalized :class:`newton.Model` is
flat arrays of bodies, shapes and DOFs -- so everything the pick controller will
later need to address has to be recorded while the model is being built.  That is
what :class:`ChessWorld` is: the index bookkeeping for one replica of the scene.

Build order inside a world is deliberate.  **The arm goes in first**, before any
piece, for two reasons:

* the arm is a chain of 1-DOF joints, so as long as nothing with a free joint
  precedes it its *coordinate* indices and its *DOF* indices coincide -- pieces
  are free bodies with 7 coordinates and 6 DOFs each, and putting them first
  would desynchronise ``joint_q`` from ``joint_qd``/``joint_target_q``;
* the arm's body and DOF indices then match those of a robot-only model built
  with the same :func:`~robochess_newton.robots.load_robot` call, which is what
  ``newton.ik`` needs (an IK solve over the full scene model would happily
  "solve" by teleporting the chess pieces, and ``joint_dof_mask`` only exists on
  newton 1.6).

Multi-world uses both of newton's mechanisms, but not symmetrically.  Identical
worlds are stamped out with :meth:`newton.ModelBuilder.replicate`, which is what
``SolverMuJoCo`` wants: it converts one world to a MuJoCo model and instances it,
so it *requires* every newton world to hold the same bodies, joints and shapes.
Worlds that differ -- two arms compared side by side, two scenarios at once --
are therefore assembled with plain ``add_builder`` calls into a **single** newton
world at the offsets ``replicate`` would have used
(:func:`newton.utils.compute_world_offsets` reproduces its centred grid exactly).
:class:`ChessWorld` indexes scene replicas either way; only ``model.world_count``
tells them apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

# board_layout ahead of everything else on purpose: importing it installs the package's
# two import-time guards -- PXR_WORK_THREAD_LIMIT, which openusd's UsdPhysics parser
# needs before the process' first pxr import, and the sys.path shield that keeps this
# repo's own newton/ directory from shadowing the real package.
from . import board_layout as bl
from .assets import PieceAssets, add_board, add_table, contact_budget
from .board_layout import BoardLayout, Vec3
from .robots import NewtonRobotSpec, RobotHandle, get_spec, joint_target_positions, load_robot

import numpy as np
import warp as wp

import newton
import newton.utils

__all__ = [
    "CAMERA_ASPECT",
    "CAMERA_RADIUS_FACTOR",
    "CAMERA_TARGET_HEIGHT",
    "DEFAULT_FPS",
    "SIM_DT",
    "SOLVER_KWARGS",
    "WORLD_SPACING",
    "ChessScene",
    "ChessWorld",
    "drop_stale_custom_attributes",
    "look_at",
    "retarget_merged_actuators",
]

SIM_DT = 1.0 / 240.0
"""Physics step. 60 Hz frames therefore run 4 substeps.

Measured over all five scenarios on both newton versions: the pieces settle to
-1.7 mm and stay there (|qd| <= 0.0016). The Isaac Lab task runs 1/120 with 4x
decimation; 1/240 is the rate the Newton contact model was validated at."""

DEFAULT_FPS = 60

WORLD_SPACING = (0.0, bl.TABLE_SIZE[1] + 0.20, 0.0)
"""Offset between replicated worlds: one table width plus a gap, along ``y``.

Along ``y`` rather than ``x`` so that a row of worlds is side by side from the
arm's point of view instead of one scene standing in the next one's playing
direction."""

SOLVER_KWARGS: dict[str, Any] = {
    "solver": "newton",
    "integrator": "implicitfast",
    "iterations": 15,
    "ls_iterations": 50,
    "cone": "elliptic",
    "impratio": 50.0,
    "use_mujoco_contacts": False,
}
"""SolverMuJoCo configuration validated on 6-32 pieces, 1-8 worlds.

``use_mujoco_contacts=False`` is not a preference: with the MJWarp-internal
contact path *and* replicated worlds, only world 0 collides and worlds 1..N-1
free-fall (measured dz = -311.67 mm), whatever ``nconmax`` and ``separate_worlds``
are set to. ``impratio=50`` with the elliptic cone is what keeps a 25 g pawn from
sliding out from between the fingers."""

CAMERA_RADIUS_FACTOR = 3.2
"""Camera distance as a multiple of the scene's larger half-extent.

Tuned by looking at the rendered frames: 2.4 fits the table but clips the top of a
franka standing at its home pose, which reaches 0.66 m above the table top."""

CAMERA_ASPECT = 16.0 / 9.0
"""Frame aspect the default camera assumes, matching ``add_viewer_args``' 1280x720."""

CAMERA_TARGET_HEIGHT = 0.23
"""Height of the camera's look-at point above the table top.

Halfway up the tallest arm rather than at the board, so a three-quarter view holds
both the pieces and the elbow."""


def look_at(eye: Sequence[float], target: Sequence[float]) -> tuple[float, float]:
    """``(pitch, yaw)`` in degrees for a newton camera at *eye* looking at *target*.

    ``Camera._set_orientation_from_direction`` derives the angles from the *view*
    direction as ``pitch = asin(d.z)``, ``yaw = atan2(d.y, d.x)``.  Getting the yaw
    sign wrong points the camera exactly away from the scene and renders an empty
    floor, which is a slow thing to debug from PNGs, so this is a function rather
    than a comment.
    """
    direction = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    distance = float(np.linalg.norm(direction))
    if distance == 0.0:
        raise ValueError("camera eye and target coincide")
    return math.degrees(math.asin(direction[2] / distance)), math.degrees(
        math.atan2(direction[1], direction[0])
    )


def _broadcast(values: Sequence[Any], count: int, name: str) -> list[Any]:
    """Stretch a one-entry setting over every world, or check a per-world sequence."""
    values = list(values)
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} has {len(values)} entries but the scene has {count} worlds")
    return values


def drop_stale_custom_attributes(builder: newton.ModelBuilder) -> dict[str, list[int]]:
    """Delete custom-attribute entries that address entities the builder does not have.

    ``add_mjcf(..., collapse_fixed_joints=True)`` keys the MuJoCo custom attributes
    by the body index the file had *before* the collapse and never renumbers them,
    so an arm whose asset both authors ``gravcomp`` and has fixed-jointed links
    leaves values sitting on indices past ``builder.body_count``.  Alone in a
    builder that is invisible -- the dense array is only ``body_count`` long, so the
    extra keys are dropped.  Add pieces afterwards and those indices become real
    bodies, which silently inherit the arm's settings.

    Measured on Menagerie ``i2rt_yam/yam.xml`` (5 stale ``mujoco:gravcomp`` keys
    8..12, identical on newton 1.2.1 and 1.6.0.dev0; franka/piper/rebot have none):
    the first five chess pieces got ``gravcomp = 1``, i.e. their weight was cancelled
    inside MuJoCo, and they hung 2 mm above the board for the whole run with
    ``|qd| == 0`` exactly.  Nothing about that reads as an asset bug from the
    outside -- the pieces simply never settle.

    Called on a builder that holds only the arm, so anything out of range is stale
    by construction.  Returns the dropped keys per attribute, for the scripts to
    report.
    """
    frequency = newton.Model.AttributeFrequency
    counts = {
        frequency.BODY: builder.body_count,
        frequency.JOINT: builder.joint_count,
        frequency.JOINT_DOF: builder.joint_dof_count,
        frequency.JOINT_COORD: builder.joint_coord_count,
        frequency.SHAPE: builder.shape_count,
    }
    dropped: dict[str, list[int]] = {}
    for name, attribute in builder.custom_attributes.items():
        count = counts.get(attribute.frequency)
        # String frequencies (mujoco:actuator, mujoco:tendon, ...) carry lists, not
        # index dicts, and are appended to rather than addressed, so they cannot go
        # stale this way.
        if count is None or not isinstance(attribute.values, dict):
            continue
        stale = sorted(key for key in attribute.values if key >= count)
        if stale:
            for key in stale:
                del attribute.values[key]
            dropped[name] = stale
    return dropped


def retarget_merged_actuators(
    builder: newton.ModelBuilder, source: newton.ModelBuilder, first_row: int, dof_offset: int
) -> int:
    """Point an appended arm's imported MuJoCo actuators back at its own DOFs.

    ``mujoco:actuator_trnid`` stores the **DOF index** an MJCF actuator drives, and
    ``ModelBuilder.add_builder`` copies the rows across without adding the
    destination's DOF offset (the value is a ``wp.vec2i``, which the merge's scalar
    remapping does not touch).  newton 1.6 reads those rows to give each joint-target
    actuator the asset's ``forcerange``/``ctrlrange``, so the second arm's torque
    limits land on the first arm's joints.

    Measured, ``ChessScene(robot=["franka", "yam"])`` on newton 1.6.0.dev0: yam's
    seven rows kept ``trnid = 0..6``, so franka's joints 1..7 were clamped to yam's
    +-28/+-10 Nm instead of their own 87/12 Nm.  With ``jnt_actgravcomp`` routing
    gravity compensation through the actuators, that clamp eats the compensation and
    the arm hangs at ``q4 = -1.571`` instead of its ``-2.200`` home -- and only when
    yam is *not* world 0, which makes it look like a franka problem.

    A no-op on newton 1.2.1, which ignores the imported ranges altogether (all
    ``actuator_forcerange`` rows come out ``(0, 0)``, i.e. unlimited, for every arm).
    Skips rows that the merge already remapped, so it stays correct if a future
    newton fixes this.  Returns the number of rows it retargeted.

    Only the ``add_builder`` path calls this, and the asymmetry is deliberate.
    ``replicate`` leaves the same stale values behind -- after
    ``builder.replicate(yam_world, 3)`` all three copies still read ``trnid = 0..6`` --
    but ``SolverMuJoCo`` consumes only the actuator rows whose ``actuator_world`` is the
    template world and skips the rest, so nothing ever reads the copies.
    """
    if dof_offset == 0:
        return 0
    attribute = builder.custom_attributes.get("mujoco:actuator_trnid")
    origin = source.custom_attributes.get("mujoco:actuator_trnid")
    if attribute is None or origin is None or not origin.values:
        return 0

    transmission = builder.custom_attributes.get("mujoco:actuator_trntype")
    joint_transmissions = (0, 1)  # TrnType.JOINT, TrnType.JOINT_IN_PARENT
    retargeted = 0
    for index, source_value in enumerate(origin.values):
        row = first_row + index
        if row >= len(attribute.values):
            break
        value = attribute.values[row]
        if value is None or int(value[0]) < 0 or int(value[0]) != int(source_value[0]):
            continue
        if transmission is not None and transmission.values and row < len(transmission.values):
            kind = transmission.values[row]
            if kind is not None and int(kind) not in joint_transmissions:
                continue
        attribute.values[row] = wp.vec2i(int(value[0]) + dof_offset, int(value[1]))
        retargeted += 1
    return retargeted


@dataclass(frozen=True)
class ChessWorld:
    """Where one replica of the scene landed in the finalized model.

    Every index is absolute -- it addresses :class:`newton.Model` arrays directly.
    Positions returned by the methods are in **world** coordinates, i.e. the Isaac
    Lab environment-frame value plus :attr:`origin`; :attr:`board_center` is the
    environment-frame value, so it can be compared with
    ``ChessPickEnvCfg.board_center()`` directly (it depends only on the arm, so it
    is the same in every world of a homogeneous scene).
    """

    index: int
    spec: NewtonRobotSpec
    layout: BoardLayout
    board_center: tuple[float, float]
    table_top_z: float
    origin: Vec3

    robot: RobotHandle
    """Arm indices, already shifted into this world."""

    piece_bodies: tuple[int, ...]
    """Body index of each entry of ``layout.pieces``, in that order."""

    piece_coords: tuple[int, ...]
    """First ``joint_q`` coordinate of each piece's free joint (7 per piece:
    ``xyz`` then ``xyzw``). ``joint_qd`` uses 6 per piece, so this is *not* the
    piece's DOF index."""

    body_start: int
    body_count: int
    joint_start: int
    joint_count: int
    dof_start: int
    dof_count: int
    coord_start: int
    coord_count: int

    arm_coords: tuple[int, ...]
    """``joint_q`` indices of the arm's DOFs. Equal to ``robot.arm_dofs`` only in
    world 0; later worlds sit behind this world's free joints, whose coordinate and
    DOF counts differ."""

    gripper_coords: tuple[int, ...]

    @property
    def piece_names(self) -> tuple[str, ...]:
        return tuple(piece.name for piece in self.layout.pieces)

    @property
    def piece_kinds(self) -> tuple[str, ...]:
        return tuple(piece.kind for piece in self.layout.pieces)

    @property
    def piece_count(self) -> int:
        return len(self.layout.pieces)

    @property
    def ee_body(self) -> int:
        return self.robot.ee_body

    @property
    def arm_dofs(self) -> tuple[int, ...]:
        return self.robot.arm_dofs

    @property
    def gripper_dofs(self) -> tuple[int, ...]:
        return self.robot.gripper_dofs

    @property
    def board_center_world(self) -> tuple[float, float]:
        return self.board_center[0] + self.origin[0], self.board_center[1] + self.origin[1]

    def piece_body(self, piece: int | str) -> int:
        """Body index of a piece, by position in ``layout.pieces`` or by name."""
        if isinstance(piece, int):
            return self.piece_bodies[piece]
        return self.piece_bodies[self.piece_names.index(piece)]

    def kind_of_body(self, body: int) -> str:
        """Piece kind of a body index. Raises for a body that is not a piece."""
        return self.piece_kinds[self.piece_bodies.index(body)]

    def square_position(self, file: int, rank: int) -> Vec3:
        """World resting position of a piece standing on square ``(file, rank)``."""
        x, y, z = bl.board_square_pos(self.layout, file, rank, self.board_center, self.table_top_z)
        return x + self.origin[0], y + self.origin[1], z + self.origin[2]

    def start_positions(self) -> list[Vec3]:
        """World start position of every piece, in ``layout.pieces`` order."""
        return [self.square_position(piece.file, piece.rank) for piece in self.layout.pieces]

    def free_square_positions(self) -> list[Vec3]:
        return [self.square_position(file, rank) for file, rank in self.layout.free_squares()]

    def tray_positions(self) -> list[Vec3]:
        """World position of every capture-tray slot, in the lab's column-major order."""
        return [
            (x + self.origin[0], y + self.origin[1], self.table_top_z + self.origin[2])
            for x, y in bl.capture_tray_slots(self.layout, self.board_center)
        ]

    def target_positions(self) -> list[Vec3]:
        """Destinations that are both within reach and safely inside the table edge.

        Mirrors ``ChessPickEnvCfg._target_positions``.  Returns an empty list rather
        than raising when an arm cannot reach any square of a scenario -- that is a
        legitimate thing to *visualize*, and only the pick controller has to refuse
        it."""
        candidates = self.free_square_positions() + self.tray_positions()
        return [
            position
            for position in candidates
            if self.distance_from_base(position) <= self.spec.reach and self.is_on_table(position)
        ]

    def reachable_piece_indices(self) -> list[int]:
        """Indices into ``layout.pieces`` whose start square the arm can reach."""
        return [
            index
            for index, position in enumerate(self.start_positions())
            if self.distance_from_base(position) <= self.spec.reach
        ]

    def distance_from_base(self, position: Sequence[float]) -> float:
        """Planar distance from the arm's base to a world position."""
        base_x = self.spec.base_pos[0] + self.origin[0]
        base_y = self.spec.base_pos[1] + self.origin[1]
        return math.hypot(position[0] - base_x, position[1] - base_y)

    def is_on_table(self, position: Sequence[float]) -> bool:
        return bl.is_on_table(position[0] - self.origin[0], position[1] - self.origin[1])


class ChessScene:
    """Table, board, pieces and one arm per world, ready for ``SolverMuJoCo``.

    ``robot``, ``scenario`` and ``board_scale`` accept either a single value --
    every world identical, built with :meth:`newton.ModelBuilder.replicate` -- or a
    per-world sequence, in which case the replicas are laid out side by side inside
    one newton world because ``SolverMuJoCo`` only accepts homogeneous worlds.

    Construction only fills a :class:`newton.ModelBuilder`; :meth:`finalize` turns
    that into the model, solver, collision pipeline and states.  The two are
    separate so a caller can inspect or extend the builder, and because only one
    ``Model`` + ``SolverMuJoCo`` pair may exist per process (two segfault).
    """

    def __init__(
        self,
        robot: str | NewtonRobotSpec | Sequence[str | NewtonRobotSpec] = "franka",
        scenario: str | Sequence[str] = "4x4",
        *,
        world_count: int | None = None,
        board_scale: float | None | Sequence[float | None] = None,
        visual: bool = True,
        world_spacing: Sequence[float] = WORLD_SPACING,
        table_top_z: float = bl.TABLE_TOP_Z,
        assets: PieceAssets | None = None,
        fps: int = DEFAULT_FPS,
        sim_dt: float = SIM_DT,
    ) -> None:
        robots = robot if isinstance(robot, (list, tuple)) else [robot]
        scenarios = scenario if isinstance(scenario, (list, tuple)) else [scenario]
        scales = board_scale if isinstance(board_scale, (list, tuple)) else [board_scale]
        count = world_count or max(len(robots), len(scenarios), len(scales))
        if count < 1:
            raise ValueError(f"world_count must be >= 1, got {count}")

        specs = [get_spec(entry) for entry in _broadcast(robots, count, "robot")]
        names = _broadcast(scenarios, count, "scenario")
        board_scales = _broadcast(scales, count, "board_scale")

        self.visual = visual
        self.table_top_z = table_top_z
        self.world_spacing = tuple(float(v) for v in world_spacing)
        self.assets = assets or PieceAssets()
        self.fps = fps
        self.sim_dt = sim_dt
        self.frame_dt = 1.0 / fps
        self.sim_substeps = max(1, round(self.frame_dt / sim_dt))
        self.sim_time = 0.0
        self.frame_index = 0
        self.dropped_attributes: dict[str, list[int]] = {}
        """Stale asset attribute keys healed at build time, unioned over every world
        builder. Two worlds of different arms drop different keys of the same attribute,
        so this accumulates rather than replaces; see :func:`drop_stale_custom_attributes`."""
        self.retargeted_actuators = 0

        self.homogeneous = len({(s.key, n, b) for s, n, b in zip(specs, names, board_scales)}) == 1
        offsets = [
            tuple(float(v) for v in offset)
            for offset in newton.utils.compute_world_offsets(count, self.world_spacing, newton.Axis.Z)
        ]

        self.builder = newton.ModelBuilder(up_axis="Z")
        # Must run before the builder's first add_*: the MuJoCo schema attributes of
        # a USD asset are only picked up if the attribute set is already registered.
        newton.solvers.SolverMuJoCo.register_custom_attributes(self.builder)

        worlds: list[ChessWorld] = []
        if self.homogeneous:
            world_builder, template = self._build_world(specs[0], names[0], board_scales[0])
            self.builder.replicate(world_builder, count, spacing=self.world_spacing)
            for index in range(count):
                worlds.append(
                    self._shift(
                        template,
                        index,
                        offsets[index],
                        body=index * world_builder.body_count,
                        joint=index * world_builder.joint_count,
                        dof=index * world_builder.joint_dof_count,
                        coord=index * world_builder.joint_coord_count,
                        counts=(
                            world_builder.body_count,
                            world_builder.joint_count,
                            world_builder.joint_dof_count,
                            world_builder.joint_coord_count,
                        ),
                    )
                )
        else:
            for index in range(count):
                world_builder, template = self._build_world(specs[index], names[index], board_scales[index])
                body, joint = self.builder.body_count, self.builder.joint_count
                dof, coord = self.builder.joint_dof_count, self.builder.joint_coord_count
                # No begin_world()/end_world() around this: SolverMuJoCo builds ONE
                # MuJoCo model and instances it per newton world, so it rejects a
                # multi-world model whose worlds differ ("SolverMuJoCo requires
                # homogeneous worlds. World 0 has 20 bodies, but world 1 has 14")
                # and refuses separate_worlds=False for world_count > 1 -- the same
                # on newton 1.2.1 and 1.6.0.dev0. Differing scenes therefore go side
                # by side into a single newton world, at the offsets replicate would
                # have used. Everything except model.world_count behaves the same.
                actuator_rows = self._actuator_row_count()
                self.builder.add_builder(
                    world_builder, xform=wp.transform(wp.vec3(*offsets[index]), wp.quat_identity())
                )
                self.retargeted_actuators += retarget_merged_actuators(
                    self.builder, world_builder, actuator_rows, dof
                )
                worlds.append(
                    self._shift(
                        template,
                        index,
                        offsets[index],
                        body=body,
                        joint=joint,
                        dof=dof,
                        coord=coord,
                        counts=(
                            world_builder.body_count,
                            world_builder.joint_count,
                            world_builder.joint_dof_count,
                            world_builder.joint_coord_count,
                        ),
                    )
                )

        # One ground plane for the whole scene rather than one per world: added on
        # the scene builder itself it gets shape_world = -1, which collides in every
        # world and avoids N coincident, z-fighting ground quads in the viewer. It
        # also keeps SolverMuJoCo's homogeneity check happy, which counts only
        # shapes with world >= 0.
        self.ground = self.builder.add_ground_plane()

        self.worlds: tuple[ChessWorld, ...] = tuple(worlds)
        self.world_count = count
        """Number of scene replicas. ``model.world_count`` is 1 when they differ."""
        self.world_offsets = tuple(offsets)
        self.contact_max = contact_budget(sum(world.piece_count for world in self.worlds))

        self.model: newton.Model | None = None
        self.solver: newton.solvers.SolverMuJoCo | None = None
        self.collision_pipeline: newton.CollisionPipeline | None = None
        self.contacts: Any = None
        self.state_0: newton.State | None = None
        self.state_1: newton.State | None = None
        self.control: newton.Control | None = None
        self.viewer: Any = None
        self._initial_joint_q: np.ndarray | None = None

    ##
    # Build
    ##

    def _build_world(
        self, spec: NewtonRobotSpec, scenario: str, board_scale: float | None
    ) -> tuple[newton.ModelBuilder, dict[str, Any]]:
        """Assemble one world into a fresh builder and record its local indices."""
        layout = bl.make_layout(scenario, bl.board_scale_for(scenario, board_scale))
        # ChessPickEnvCfg.board_center(): board_distance in front of the arm's base.
        board_center = (spec.base_pos[0] + spec.board_distance, spec.base_pos[1])

        builder = newton.ModelBuilder(up_axis="Z")
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # The arm first, so its coordinate indices equal its DOF indices; see the
        # module docstring. Free joints appear only after it.
        handle = load_robot(builder, spec)
        for name, stale in drop_stale_custom_attributes(builder).items():
            merged = set(self.dropped_attributes.get(name, ())) | set(stale)
            self.dropped_attributes[name] = sorted(merged)
        if builder.joint_coord_count != builder.joint_dof_count:
            raise RuntimeError(
                f"{spec.key} contributed {builder.joint_coord_count} coordinates for "
                f"{builder.joint_dof_count} DOFs; ChessScene assumes a chain of 1-DOF joints"
            )
        home = np.asarray(builder.joint_q, dtype=np.float32)
        handle.write_joint_positions(home, opened=True)
        builder.joint_q[:] = home.tolist()

        add_table(builder, table_top_z=self.table_top_z, visual=self.visual)
        add_board(builder, layout, board_center, self.table_top_z, visual=self.visual)
        if self.visual:
            self._add_capture_tray(builder, layout, board_center)

        piece_bodies: list[int] = []
        piece_coords: list[int] = []
        for piece in layout.pieces:
            position, quat = bl.piece_world_pose(layout, piece, board_center, self.table_top_z)
            body = self.assets.add_piece(
                builder,
                piece.kind,
                wp.transform(wp.vec3(*position), wp.quat(*quat)),
                piece.color,
                visual=self.visual,
                label=piece.name,
            )
            piece_bodies.append(body)
            # add_body appends the free joint, so it is the builder's newest joint.
            piece_coords.append(builder.joint_q_start[builder.joint_count - 1])

        return builder, {
            "spec": spec,
            "layout": layout,
            "board_center": board_center,
            "handle": handle,
            "piece_bodies": tuple(piece_bodies),
            "piece_coords": tuple(piece_coords),
        }

    def _actuator_row_count(self) -> int:
        """How many ``mujoco:actuator`` rows the scene builder already holds.

        Counted on ``trnid`` because that is the row list :func:`retarget_merged_actuators`
        goes on to index; ``robots._actuator_row_count`` counts the same rows on
        ``forcerange`` for the same reason.
        """
        attribute = self.builder.custom_attributes.get("mujoco:actuator_trnid")
        return len(attribute.values) if attribute is not None and attribute.values else 0

    def _add_capture_tray(
        self, builder: newton.ModelBuilder, layout: BoardLayout, board_center: tuple[float, float]
    ) -> int:
        """The render-only slab marking where captured pieces are parked."""
        tray_x, tray_y = bl.capture_tray_center(layout, board_center)
        size = bl.capture_tray_size()
        return builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(tray_x, tray_y, self.table_top_z + size[2] / 2.0), wp.quat_identity()
            ),
            hx=size[0] / 2.0,
            hy=size[1] / 2.0,
            hz=size[2] / 2.0,
            cfg=newton.ModelBuilder.ShapeConfig(
                has_shape_collision=False, has_particle_collision=False, is_visible=True
            ),
            color=wp.vec3(*bl.CAPTURE_TRAY_COLOR),
            label="capture_tray",
        )

    def _shift(
        self,
        template: dict[str, Any],
        index: int,
        origin: Vec3,
        *,
        body: int,
        joint: int,
        dof: int,
        coord: int,
        counts: tuple[int, int, int, int],
    ) -> ChessWorld:
        """Turn one world's builder-local indices into scene-absolute ones."""
        handle: RobotHandle = template["handle"]
        shifted = handle.shifted(body, joint, dof)
        return ChessWorld(
            index=index,
            spec=template["spec"],
            layout=template["layout"],
            board_center=template["board_center"],
            table_top_z=self.table_top_z,
            origin=origin,
            robot=shifted,
            piece_bodies=tuple(body + local for local in template["piece_bodies"]),
            piece_coords=tuple(coord + local for local in template["piece_coords"]),
            body_start=body,
            body_count=counts[0],
            joint_start=joint,
            joint_count=counts[1],
            dof_start=dof,
            dof_count=counts[2],
            coord_start=coord,
            coord_count=counts[3],
            # The arm is the first thing in a world and is all 1-DOF joints, so its
            # coordinate offsets inside the world equal its DOF offsets.
            arm_coords=tuple(coord + local for local in handle.arm_dofs),
            gripper_coords=tuple(coord + local for local in handle.gripper_dofs),
        )

    ##
    # Finalize / simulate
    ##

    def finalize(self, device: str | None = None) -> newton.Model:
        """Build the model, solver, collision pipeline and states.

        ``rigid_contact_max`` is a hard failure mode rather than a performance knob:
        undersized, the pipeline silently drops *every* contact and the pieces free
        fall (measured dz = -311.67 mm after 60 frames).
        """
        self.model = self.builder.finalize(device=device)
        self.model.rigid_contact_max = self.contact_max
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=self.contact_max,
            broad_phase="nxn",
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            nconmax=self.contact_max,
            njmax=2 * self.contact_max,
            **SOLVER_KWARGS,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._initial_joint_q = self.model.joint_q.numpy().copy()
        self.apply_home_targets()
        self.reset()
        return self.model

    def _require_model(self) -> newton.Model:
        if self.model is None:
            raise RuntimeError("ChessScene.finalize() has not been called yet")
        return self.model

    def apply_home_targets(self) -> None:
        """Command every arm to its home posture with the gripper open.

        The importers leave the drive targets at zero, which for most of these arms
        means folded flat through the table, so this has to happen before the first
        step, not merely at reset.
        """
        self._require_model()
        targets = joint_target_positions(self.control)
        values = targets.numpy()
        for world in self.worlds:
            world.robot.write_joint_positions(values, opened=True)
        targets.assign(values)

    def reset(self) -> None:
        """Put every joint back to its initial value and re-run forward kinematics."""
        model = self._require_model()
        self.state_0.joint_q.assign(self._initial_joint_q)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time = 0.0
        self.frame_index = 0

    def substep(self) -> None:
        """One physics step of :data:`SIM_DT`."""
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        """One rendered frame's worth of physics."""
        self._require_model()
        for _ in range(self.sim_substeps):
            self.substep()
        self.frame_index += 1
        # Derived from the frame counter, not accumulated: ViewerUSD turns the time
        # into a timecode with int(time * fps) and an accumulated 1/60 loses frames.
        self.sim_time = self.frame_index * self.frame_dt

    ##
    # Viewer
    ##

    def default_camera(self, zoom: float = 1.0) -> tuple[Vec3, Vec3]:
        """A ``(eye, target)`` pair that frames every world's board and arm."""
        xs = [world.board_center_world[0] for world in self.worlds]
        ys = [world.board_center_world[1] for world in self.worlds]
        target = (
            (min(xs) + max(xs)) / 2.0,
            (min(ys) + max(ys)) / 2.0,
            self.table_top_z + CAMERA_TARGET_HEIGHT,
        )
        half_x = bl.TABLE_SIZE[0] / 2.0 + (max(xs) - min(xs)) / 2.0
        half_y = bl.TABLE_SIZE[1] / 2.0 + (max(ys) - min(ys)) / 2.0
        # A row of worlds along y lands across the frame's long axis, so it needs
        # less distance than the same extent along x. Only sqrt(aspect) of it,
        # though: the row also spreads in depth, and dividing by the full 16/9 cuts
        # the outer two of four worlds off the frame (looked at the PNGs).
        radius = CAMERA_RADIUS_FACTOR * max(half_x, half_y / math.sqrt(CAMERA_ASPECT)) / max(zoom, 1e-3)

        # A three-quarter view reads best for one world, but with several worlds in a
        # row along y the near one stands in front of the far ones, so swing towards
        # a frontal view as the row grows.
        frontal = min(1.0, (max(ys) - min(ys)) / (2.0 * bl.TABLE_SIZE[1]))
        direction = np.array([-(0.55 + 0.40 * frontal), -0.66 * (1.0 - frontal), 0.55])
        direction /= np.linalg.norm(direction)
        eye = tuple(float(t + radius * d) for t, d in zip(target, direction))
        return eye, target

    def attach_viewer(
        self,
        viewer: Any,
        *,
        eye: Sequence[float] | None = None,
        target: Sequence[float] | None = None,
        zoom: float = 1.0,
    ) -> None:
        """Hand the model to a viewer and aim its camera at the scene."""
        model = self._require_model()
        # No max_worlds=: removed in newton 1.6.
        viewer.set_model(model)
        # A viewer with no explicit spacing spreads the worlds itself, on a square
        # grid with ceil(1.5 * max extent) between them, on top of wherever the model
        # actually puts them (ViewerBase._auto_compute_world_offsets, both versions).
        # Our worlds are already a row along y, so the display grid turned four
        # tables into a 2x2 block standing 4 m apart -- render positions that no
        # longer matched the physics. Zero it and show the model as built.
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        default_eye, default_target = self.default_camera(zoom)
        eye = tuple(eye) if eye is not None else default_eye
        target = tuple(target) if target is not None else default_target
        pitch, yaw = look_at(eye, target)
        viewer.set_camera(pos=wp.vec3(*eye), pitch=pitch, yaw=yaw)
        self.viewer = viewer

    def render(self) -> None:
        """Log the current state to the attached viewer."""
        if self.viewer is None:
            raise RuntimeError("ChessScene.attach_viewer() has not been called yet")
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    ##
    # Introspection
    ##

    @property
    def bodies_per_world(self) -> int:
        """Only defined when every world is identical."""
        return self._per_world("body_count")

    @property
    def joints_per_world(self) -> int:
        return self._per_world("joint_count")

    @property
    def dofs_per_world(self) -> int:
        return self._per_world("dof_count")

    @property
    def coords_per_world(self) -> int:
        return self._per_world("coord_count")

    def _per_world(self, field: str) -> int:
        if not self.homogeneous:
            raise RuntimeError(
                f"{field} differs between worlds; read it off ChessScene.worlds[i] instead"
            )
        return getattr(self.worlds[0], field)

    def body_index(self, world: int, local_body: int) -> int:
        """Absolute body index of a world-local body index."""
        return self.worlds[world].body_start + local_body

    def piece_body_ids(self) -> np.ndarray:
        """Body index of every piece in every world, shape ``(world_count, n_pieces)``.

        Only for homogeneous scenes; heterogeneous worlds have different piece counts.
        """
        self._per_world("body_count")
        return np.array([world.piece_bodies for world in self.worlds], dtype=np.int32)

    def robot_only_builder(self, world: int = 0) -> tuple[newton.ModelBuilder, RobotHandle]:
        """A fresh builder holding just this world's arm, for ``newton.ik``.

        An IK solve over the scene model would also drive the pieces' free joints
        (``joint_dof_mask`` is newton 1.6 only), so the IK model has to be the arm
        alone.  Because the arm is built first in a world, the returned handle's
        indices match this world's minus ``body_start``/``dof_start``.
        """
        spec = self.worlds[world].spec
        builder = newton.ModelBuilder(up_axis="Z")
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        handle = load_robot(builder, spec)
        drop_stale_custom_attributes(builder)
        home = np.asarray(builder.joint_q, dtype=np.float32)
        handle.write_joint_positions(home, opened=True)
        builder.joint_q[:] = home.tolist()
        return builder, handle

    def describe(self) -> str:
        """One line per world plus a totals line, for the scripts to print."""
        lines = []
        for world in self.worlds:
            center = world.board_center_world
            lines.append(
                f"  world {world.index}: {world.spec.key:<12s} {world.layout.scenario:<6s} "
                f"board={world.layout.board_usd:<20s} scale={world.layout.board_scale:.2f} "
                f"center=({center[0]:+.3f},{center[1]:+.3f}) pieces={world.piece_count:2d} "
                f"bodies={world.body_start}..{world.body_start + world.body_count - 1} "
                f"ee={world.ee_body} arm_dofs={world.arm_dofs[0]}..{world.arm_dofs[-1]} "
                f"reachable={len(world.reachable_piece_indices())}/{world.piece_count} "
                f"targets={len(world.target_positions())}"
            )
        model = self.model
        totals = (
            f"  total: worlds={self.world_count} bodies={self.builder.body_count} "
            f"shapes={self.builder.shape_count} dofs={self.builder.joint_dof_count} "
            f"contacts_max={self.contact_max} visual={self.visual}"
        )
        if model is not None:
            totals += f" newton_worlds={model.world_count} device={model.device}"
        if self.dropped_attributes:
            lines.append(
                "  dropped stale asset attributes: "
                + ", ".join(f"{name}{keys}" for name, keys in self.dropped_attributes.items())
            )
        if self.retargeted_actuators:
            lines.append(f"  retargeted {self.retargeted_actuators} imported MuJoCo actuator rows")
        return "\n".join([*lines, totals])
