"""The RoboChess table scene, assembled into a :class:`genesis.Scene`.

This is the Genesis counterpart of ``FrankaChessSceneCfg`` + ``configure_chess_scene`` +
``ChessPickEnvCfg.set_robot`` on the Isaac Lab side, and of
``robochess_newton.scene.ChessScene`` on the Newton side: ground plane, table, chess board,
one free rigid body per piece, the capture tray and one arm, with the board placed
``spec.board_distance`` in front of that arm exactly as ``ChessPickEnvCfg.board_center()``
does.

**What a "world" is here.** Genesis has one batching mechanism -- ``scene.build(n_envs=N)``
-- and it batches *the same scene* N times.  Every env holds the same entities, and each
env's state lives in its own copy of the solver arrays **in the same coordinate frame**:
``env_spacing`` moves the envs apart for *rendering* only (``scene.envs_offset``), and
``get_links_pos`` returns identical numbers for every env of an identical scene (measured).
That is the opposite of Newton, where replicated worlds are genuinely offset inside one
shared coordinate frame.

So :class:`ChessWorld` is a pair -- *which env* and *which entities* -- and the two
multi-world modes fill it in differently:

* **identical worlds** (one arm, one scenario) become ``n_envs = world_count`` real Genesis
  envs.  Every world shares the same entity objects and differs only in
  :attr:`ChessWorld.env`, and every world's :attr:`ChessWorld.origin` is the zero vector.
* **worlds that differ** (``--robot franka,yam``) cannot: a batched env is a copy, not a
  variant.  They are laid out side by side inside a **single** env at the offsets the batch
  would have used, which is exactly the fallback the Newton port makes for the same reason
  (``SolverMuJoCo`` refuses heterogeneous worlds).  Each world then has its own entities,
  ``env = 0``, and a non-zero :attr:`ChessWorld.origin`.

``describe()`` prints ``genesis_envs=1`` when that happens, so a run says which mode it is
in.  Everything downstream -- the pick task, the reporting -- indexes worlds and never looks
at the difference, because the slot bookkeeping below resolves it.

**Reading state.** Genesis' per-entity getters (``entity.get_pos()``) are one kernel launch
each, and a 32-piece board would be 33 launches per control tick.  The scene therefore
gathers every link it will ever need into one solver-global index array at construction time
and reads them all in a single ``rigid_solver.get_links_pos`` / ``_quat`` / ``_vel`` call;
the same for DOFs and for the pieces' free-joint coordinates.  :class:`ChessWorld` records
the *slots* into those arrays rather than raw indices, which is also what makes the two
multi-world modes interchangeable to callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

# board_layout first: it installs PXR_WORK_THREAD_LIMIT before anything can import pxr, and
# both the piece transcoder and the Piper USD go through pxr.
from . import board_layout as bl
from .assets import (
    BOARD_FRICTION,
    DEFAULT_VISUAL_FACES,
    PIECE_FRICTION,
    TABLE_FRICTION,
    PieceAssets,
    board_placement,
    collision_pair_budget,
    piece_color,
    table_placement,
)
from .board_layout import BoardLayout, Vec3
from .gsmath import as_numpy, from_gs_quat, to_gs_quat
from .robots import GenesisRobotSpec, RobotHandle, apply_drive_settings, get_spec, load_robot

import numpy as np

__all__ = [
    "CAMERA_ASPECT",
    "CAMERA_RADIUS_FACTOR",
    "CAMERA_TARGET_HEIGHT",
    "DEFAULT_FPS",
    "RIGID_OPTIONS",
    "SIM_DT",
    "WORLD_SPACING",
    "CameraSpec",
    "ChessScene",
    "ChessWorld",
    "init_genesis",
    "world_offsets",
]

SIM_DT = 1.0 / 240.0
"""Physics step [s]. 60 Hz frames therefore run 4 substeps.

The same rate the Newton port validated its contact model at, and it reproduces the same
settling here: measured over all five scenarios, the pieces come to rest 1.24-1.65 mm below
where they spawn and stay there, against Newton's 1.25-1.67 mm band."""

DEFAULT_FPS = 60

WORLD_SPACING = (0.0, bl.TABLE_SIZE[1] + 0.20, 0.0)
"""Offset between worlds: one table width plus a gap, along ``y``.

Along ``y`` rather than ``x`` so that a row of worlds is side by side from the arm's point
of view instead of one scene standing in the next one's playing direction.  Used both as
Genesis' ``env_spacing`` (where it only separates the *render*) and as the real offset
between side-by-side heterogeneous worlds."""

RIGID_OPTIONS: dict[str, Any] = {
    "iterations": 20,
    "ls_iterations": 20,
    "use_contact_island": False,
    "box_box_detection": True,
    "enable_multi_contact": True,
}
"""``gs.options.RigidOptions`` overrides, measured on 6-32 pieces over 1-8 envs.

Genesis' defaults are ``iterations=50, ls_iterations=50, use_contact_island=True``.  On the
4x4 board over 8 envs that is 37.7 ms/frame; the settings above are **18.4 ms/frame** and
settle the pieces to the same 1.24-1.65 mm, which is the check that the speed was not bought
by loosening the contact solve.  ``use_contact_island=False`` is the larger half of it
(37.7 -> 20.1 on its own): contact islands earn their bookkeeping when contacts form
separable clusters, and a chess board is one cluster -- every piece touching one board box.

``box_box_detection=True`` matters because two of the three static colliders here *are*
boxes (the table and the board), and the generic path gives a box-box pair far fewer contact
points than the dedicated one -- which is what a piece resting on a 10 mm slab needs.
``iterations``/``ls_iterations`` are left well above the point where the settling changes
(measured identical at 20 and at 50) rather than trimmed to the minimum."""

CAMERA_RADIUS_FACTOR = 3.2
"""Camera distance as a multiple of the scene's larger half-extent.

Tuned by looking at rendered frames: 2.4 fits the table but clips the top of a franka
standing at its home pose, which reaches 0.66 m above the table top."""

CAMERA_ASPECT = 16.0 / 9.0
"""Frame aspect the default camera assumes, matching the 1280x720 default resolution."""

CAMERA_TARGET_HEIGHT = 0.23
"""Height of the camera's look-at point above the table top.

Halfway up the tallest arm rather than at the board, so a three-quarter view holds both the
pieces and the elbow."""


def init_genesis(backend: str | None = None, seed: int | None = None, logging_level: str = "warning") -> None:
    """``gs.init`` once per process, idempotent.

    Genesis raises if ``init`` is called twice, and every entry point here has to work both
    as a script and as an import into a session that already initialised it.  ``backend``
    takes the names ``gs`` exposes as constants (``"gpu"``, ``"cuda"``, ``"cpu"``);
    ``None`` lets Genesis pick, which is the GPU when one is visible.
    """
    import genesis as gs

    if getattr(gs, "_initialized", False):
        return
    kwargs: dict[str, Any] = {"logging_level": logging_level}
    if backend is not None:
        resolved = getattr(gs, str(backend).lower(), None)
        if resolved is None:
            raise ValueError(f"unknown Genesis backend {backend!r}; try 'gpu', 'cuda' or 'cpu'")
        kwargs["backend"] = resolved
    if seed is not None:
        kwargs["seed"] = seed
    gs.init(**kwargs)


def world_offsets(count: int, spacing: Sequence[float]) -> list[Vec3]:
    """A centred row of *count* offsets, matching Genesis' own ``envs_offset`` layout.

    Genesis lays a batch out on a grid and centres it on the origin
    (``build(center_envs_at_origin=True)``); with ``n_envs_per_row = n_envs`` that grid is a
    single row.  Reproducing it here means the side-by-side fallback puts its worlds exactly
    where the batched path would have drawn them, so the two modes render the same.
    """
    spacing = np.asarray(spacing, dtype=float)
    return [tuple((index - (count - 1) / 2.0) * spacing) for index in range(count)]


def _backend_name() -> str:
    """The resolved Genesis backend, for the ``describe()`` line.

    ``gs.backend`` holds the *resolved* enum after ``gs.init`` -- ``gs.gpu`` is a meta
    value that becomes ``gs.cuda`` on this box -- and there is no ``scene.device``.
    """
    import genesis as gs

    return str(getattr(gs, "backend", "?")).rsplit(".", 1)[-1]


def _broadcast(values: Sequence[Any], count: int, name: str) -> list[Any]:
    """Stretch a one-entry setting over every world, or check a per-world sequence."""
    values = list(values)
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} has {len(values)} entries but the scene has {count} worlds")
    return values


@dataclass(frozen=True)
class ChessWorld:
    """One replica of the scene: which Genesis env it lives in, and which entities are it.

    Every index recorded here is a *slot* into one of :class:`ChessScene`'s gathered arrays,
    not a solver index, because that is what makes the two multi-world modes interchangeable
    -- see the module docstring.  :attr:`origin` is zero in the batched mode and the
    side-by-side offset otherwise; positions returned by the methods below always include
    it, so they are comparable across modes and against the Newton port.
    :attr:`board_center` does *not* include it: it is the environment-frame value, so it can
    be compared with ``ChessPickEnvCfg.board_center()`` directly.
    """

    index: int
    env: int
    """Genesis environment this world's state is read from."""

    origin: Vec3
    spec: GenesisRobotSpec
    layout: BoardLayout
    board_center: tuple[float, float]
    table_top_z: float

    robot: RobotHandle
    piece_entities: tuple[Any, ...]
    """One ``RigidEntity`` per entry of ``layout.pieces``, in that order."""

    ee_slot: int
    """Slot of the arm's ``ee_link`` in :attr:`ChessScene.watch_links`."""

    piece_slots: tuple[int, ...]
    """Slot of each piece's base link in :attr:`ChessScene.watch_links`."""

    arm_dof_slots: tuple[int, ...]
    """Slots of the arm DOFs in :attr:`ChessScene.watch_dofs`."""

    gripper_dof_slots: tuple[int, ...]
    piece_q_slots: tuple[int, ...]
    """Slot of the first of each piece's 7 free-joint coordinates in
    :attr:`ChessScene.watch_qs` (``xyz`` then Genesis' ``wxyz``)."""

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
    def board_center_world(self) -> tuple[float, float]:
        return self.board_center[0] + self.origin[0], self.board_center[1] + self.origin[1]

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

        Mirrors ``ChessPickEnvCfg._target_positions``.  Returns an empty list rather than
        raising when an arm cannot reach any square of a scenario -- that is a legitimate
        thing to *visualize*, and only the pick controller has to refuse it.
        """
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


@dataclass
class CameraSpec:
    """What :class:`ChessScene` needs to create its render camera, before ``build()``.

    Genesis' ``scene.add_camera`` is ``@gs.assert_unbuilt``, so the camera has to exist
    before the model does -- which is why this is a constructor argument rather than
    something a caller attaches afterwards, as it is on the Newton side.
    """

    res: tuple[int, int] = (1280, 720)
    eye: Vec3 | None = None
    target: Vec3 | None = None
    zoom: float = 1.0
    fov: float = 40.0
    debug: bool = False
    """``True`` also renders ``scene.draw_debug_*`` markers, which a normal camera skips."""


class ChessScene:
    """Table, board, pieces and one arm per world, ready to step.

    ``robot``, ``scenario`` and ``board_scale`` accept either a single value -- every world
    identical, batched into real Genesis envs -- or a per-world sequence, in which case the
    replicas are laid side by side inside one env.  See the module docstring.

    Construction fills a :class:`genesis.Scene`; :meth:`finalize` builds it, applies the
    drive settings (Genesis exposes gains only on a built solver) and puts every arm at its
    home posture.  The two are separate so a caller can add its own entities in between,
    and because a built scene cannot be added to.
    """

    def __init__(
        self,
        robot: str | GenesisRobotSpec | Sequence[str | GenesisRobotSpec] = "franka",
        scenario: str | Sequence[str] = "4x4",
        *,
        world_count: int | None = None,
        board_scale: float | None | Sequence[float | None] = None,
        visual: bool = True,
        world_spacing: Sequence[float] = WORLD_SPACING,
        table_top_z: float = bl.TABLE_TOP_Z,
        assets: PieceAssets | None = None,
        visual_faces: int = DEFAULT_VISUAL_FACES,
        fps: int = DEFAULT_FPS,
        sim_dt: float = SIM_DT,
        camera: CameraSpec | None = None,
        show_viewer: bool = False,
        rigid_options: dict[str, Any] | None = None,
        backend: str | None = None,
    ) -> None:
        import genesis as gs

        robots = robot if isinstance(robot, (list, tuple)) else [robot]
        scenarios = scenario if isinstance(scenario, (list, tuple)) else [scenario]
        scales = board_scale if isinstance(board_scale, (list, tuple)) else [board_scale]
        count = world_count or max(len(robots), len(scenarios), len(scales))
        if count < 1:
            raise ValueError(f"world_count must be >= 1, got {count}")

        specs = [get_spec(entry) for entry in _broadcast(robots, count, "robot")]
        names = _broadcast(scenarios, count, "scenario")
        board_scales = _broadcast(scales, count, "board_scale")

        init_genesis(backend=backend)

        self.visual = visual
        self.table_top_z = table_top_z
        self.world_spacing = tuple(float(v) for v in world_spacing)
        self.assets = assets or PieceAssets(visual_faces=visual_faces)
        self.fps = fps
        self.sim_dt = sim_dt
        self.frame_dt = 1.0 / fps
        self.sim_substeps = max(1, round(self.frame_dt / sim_dt))
        self.sim_time = 0.0
        self.frame_index = 0
        self.world_count = count
        self.built = False

        self.homogeneous = len({(s.key, n, b) for s, n, b in zip(specs, names, board_scales)}) == 1
        self.env_count = count if self.homogeneous else 1
        """Number of real Genesis envs. 1 when the worlds differ; see the module docstring."""

        if self.homogeneous and count > 1 and (self.world_spacing[0] or self.world_spacing[2]):
            # Genesis lays a batch out on a grid and, with n_envs_per_row = n_envs, that grid
            # is a single row along y; env_spacing's x entry is unused and there is no z entry
            # at all. Side-by-side worlds honour the full vector, so silently ignoring it here
            # would make the same flag mean two different things.
            print(
                f"[WARN] --world-spacing {self.world_spacing} has a non-zero x or z, but batched "
                f"Genesis envs are separated along y only; using "
                f"(0.0, {self.world_spacing[1]}, 0.0) for the render layout",
                flush=True,
            )
        offsets = world_offsets(count, (0.0, self.world_spacing[1], 0.0) if self.homogeneous else self.world_spacing)
        # Batched envs are separated for rendering only, so their *physics* origin is zero;
        # side-by-side worlds carry the offset for real.
        self.world_offsets = tuple((0.0, 0.0, 0.0) for _ in range(count)) if self.homogeneous else tuple(offsets)
        self.render_offsets = tuple(offsets)

        layouts = [bl.make_layout(name, bl.board_scale_for(name, scale)) for name, scale in zip(names, board_scales)]
        # Per *env*: Genesis clamps max_collision_pairs to the env's own possible-pair count
        # and allocates the contact cache as (n_possible_pairs, n_envs), so the batched mode
        # -- where every env holds one board -- must not multiply by the world count. The
        # side-by-side mode really does put every world's pieces in one env, and there the
        # sum is the right number.
        pieces_per_env = (
            len(layouts[0].pieces) if self.homogeneous else sum(len(layout.pieces) for layout in layouts)
        )
        self.contact_max = collision_pair_budget(pieces_per_env)

        options = dict(RIGID_OPTIONS)
        options.update(rigid_options or {})
        options.setdefault("max_collision_pairs", self.contact_max)

        resolution = camera.res if camera is not None else (1280, 720)
        self.gs_scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=sim_dt, substeps=1),
            rigid_options=gs.options.RigidOptions(**options),
            vis_options=gs.options.VisOptions(background_color=(0.05, 0.06, 0.08)),
            viewer_options=gs.options.ViewerOptions(res=resolution),
            show_viewer=show_viewer,
        )
        self.gs_scene.add_entity(gs.morphs.Plane())

        self._watch_links: list[int] = []
        self._watch_dofs: list[int] = []
        self._watch_qs: list[int] = []
        self._first_world: ChessWorld | None = None
        worlds = [
            self._build_world(index, specs[index], layouts[index])
            for index in range(count)
        ]
        self.worlds: tuple[ChessWorld, ...] = tuple(worlds)
        self.watch_links = np.asarray(self._watch_links, dtype=np.int64)
        """Solver-global link indices gathered once per read; see the module docstring."""

        self.watch_dofs = np.asarray(self._watch_dofs, dtype=np.int64)
        self.watch_qs = np.asarray(self._watch_qs, dtype=np.int64)

        self.camera_spec = camera
        self.camera = None
        if camera is not None:
            eye, target = self.default_camera(camera.zoom)
            self.camera = self.gs_scene.add_camera(
                res=tuple(camera.res),
                pos=tuple(camera.eye) if camera.eye is not None else eye,
                lookat=tuple(camera.target) if camera.target is not None else target,
                fov=camera.fov,
                GUI=False,
                debug=camera.debug,
            )

        self._home_dofs: np.ndarray | None = None
        self._home_qs: np.ndarray | None = None

    ##
    # Build
    ##

    def _build_world(self, index: int, spec: GenesisRobotSpec, layout: BoardLayout) -> ChessWorld:
        """Add one world's entities, or reuse world 0's when the batch makes them shared."""
        import genesis as gs

        # ChessPickEnvCfg.board_center(): board_distance in front of the arm's base.
        board_center = (spec.base_pos[0] + spec.board_distance, spec.base_pos[1])
        origin = self.world_offsets[index]

        if self.homogeneous and index > 0:
            # Every batched env holds the same entities in the same frame, so worlds 1..N-1
            # *are* world 0, read out of a different env. Nothing is added a second time.
            first = self._first_world
            return ChessWorld(
                index=index,
                env=index,
                origin=origin,
                spec=spec,
                layout=layout,
                board_center=board_center,
                table_top_z=self.table_top_z,
                robot=first.robot,
                piece_entities=first.piece_entities,
                ee_slot=first.ee_slot,
                piece_slots=first.piece_slots,
                arm_dof_slots=first.arm_dof_slots,
                gripper_dof_slots=first.gripper_dof_slots,
                piece_q_slots=first.piece_q_slots,
            )

        offset = np.asarray(origin, dtype=float)

        # The arm first, for parity with the Newton port's build order and because it reads
        # better in the entity list; unlike Newton, Genesis indexes per entity, so nothing
        # about correctness depends on it here.
        robot = load_robot(self.gs_scene, spec, tuple(np.asarray(spec.base_pos, dtype=float) + offset))

        table_pos, table_size = table_placement(table_top_z=self.table_top_z)
        self.gs_scene.add_entity(
            gs.morphs.Box(size=table_size, pos=tuple(np.asarray(table_pos) + offset), fixed=True),
            surface=gs.surfaces.Default(color=(*bl.TABLE_COLOR, 1.0)),
            material=gs.materials.Rigid(friction=TABLE_FRICTION),
        )

        board = board_placement(layout, board_center, self.table_top_z)
        self.gs_scene.add_entity(
            gs.morphs.Box(
                size=board.collider_size,
                pos=tuple(np.asarray(board.collider_position) + offset),
                quat=tuple(to_gs_quat(np.asarray(board.collider_quat_xyzw))),
                fixed=True,
                visualization=False,
            ),
            material=gs.materials.Rigid(friction=BOARD_FRICTION),
        )
        if self.visual:
            for subset in board.subsets:
                self.gs_scene.add_entity(
                    gs.morphs.Mesh(
                        file=str(subset.path),
                        pos=tuple(np.asarray(board.render_position) + offset),
                        quat=tuple(to_gs_quat(np.asarray(board.render_quat_xyzw))),
                        fixed=True,
                        collision=False,
                        convexify=False,
                    ),
                    surface=gs.surfaces.Default(color=(*subset.color, 1.0)),
                )
            self._add_capture_tray(layout, board_center, offset)

        piece_entities = []
        piece_slots = []
        piece_q_slots = []
        for piece in layout.pieces:
            position, quat = bl.piece_world_pose(layout, piece, board_center, self.table_top_z)
            entity = self.gs_scene.add_entity(
                gs.morphs.URDF(
                    file=str(self.assets.urdf_path(piece.kind)),
                    pos=tuple(np.asarray(position) + offset),
                    quat=tuple(to_gs_quat(np.asarray(quat))),
                    fixed=False,
                    # The URDF carries the authored mass and the inertia tensor computed
                    # from the hulls; recomputing would integrate 16 *overlapping* hulls and
                    # inflate the mass by 16 %. See PieceAssets.mass_properties.
                    recompute_inertia=False,
                    merge_fixed_links=False,
                    # align=False keeps the piece's AUTHORED frame -- base plane at z=0,
                    # +Z up the piece axis. Genesis' default (align=None -> on for a plain
                    # rigid body) re-origins a free body's link frame at its centre of mass
                    # and rotates it onto the inertia principal axes, which for a pawn is
                    # +24.3 mm of z and a 24.8-degree spin. Nothing announces it: the free
                    # joint's qpos and the solver's link poses are then in that shifted
                    # frame, so the GraspGen matrices (authored in the piece frame with
                    # z=0 on the board), the square positions and the 10 mm "on the
                    # surface" success clause would all be quietly measuring the wrong
                    # point. Measured before the fix: pieces resting on the board read back
                    # at z = 0.7947 instead of 0.7703.
                    align=False,
                    visualization=self.visual,
                ),
                surface=gs.surfaces.Default(color=piece_color(piece.color)),
                material=gs.materials.Rigid(friction=PIECE_FRICTION),
            )
            piece_entities.append(entity)
            piece_slots.append(len(self._watch_links))
            self._watch_links.append(entity.base_link.idx)
            piece_q_slots.append(len(self._watch_qs))
            self._watch_qs.extend(range(entity.q_start, entity.q_start + 7))

        ee_slot = len(self._watch_links)
        self._watch_links.append(robot.ee_link)
        arm_slots = tuple(len(self._watch_dofs) + i for i in range(len(robot.arm_dofs)))
        self._watch_dofs.extend(robot.arm_dofs)
        gripper_slots = tuple(len(self._watch_dofs) + i for i in range(len(robot.gripper_dofs)))
        self._watch_dofs.extend(robot.gripper_dofs)

        world = ChessWorld(
            index=index,
            env=0,
            origin=origin,
            spec=spec,
            layout=layout,
            board_center=board_center,
            table_top_z=self.table_top_z,
            robot=robot,
            piece_entities=tuple(piece_entities),
            ee_slot=ee_slot,
            piece_slots=tuple(piece_slots),
            arm_dof_slots=arm_slots,
            gripper_dof_slots=gripper_slots,
            piece_q_slots=tuple(piece_q_slots),
        )
        if index == 0:
            self._first_world = world
        return world

    def _add_capture_tray(
        self, layout: BoardLayout, board_center: tuple[float, float], offset: np.ndarray
    ) -> None:
        """The render-only slab marking where captured pieces are parked."""
        import genesis as gs

        tray_x, tray_y = bl.capture_tray_center(layout, board_center)
        size = bl.capture_tray_size()
        self.gs_scene.add_entity(
            gs.morphs.Box(
                size=size,
                pos=tuple(np.asarray([tray_x, tray_y, self.table_top_z + size[2] / 2.0]) + offset),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(*bl.CAPTURE_TRAY_COLOR, 1.0)),
        )

    ##
    # Finalize / simulate
    ##

    def finalize(
        self,
        arm_kp: float | None = None,
        arm_kd: float | None = None,
        gripper_kp: float | None = None,
        gripper_kd: float | None = None,
        arm_armature: float | None = None,
        gripper_armature: float | None = None,
    ) -> Any:
        """Build the Genesis scene and put every arm at its home posture.

        The drive settings are arguments here rather than a separate call because Genesis
        exposes ``set_dofs_kp`` / ``set_dofs_armature`` only on a *built* solver -- the
        opposite of Newton, where they are ``ModelBuilder`` fields and the pick task edits
        them before ``finalize``.  Passing them in keeps the "gains are chosen once, before
        anything moves" property the Newton port gets from its ordering.
        """
        if self.built:
            raise RuntimeError("ChessScene.finalize() has already been called")
        self.gs_scene.build(
            n_envs=self.env_count,
            env_spacing=(self.world_spacing[0], self.world_spacing[1]),
            # One row, so the render layout of a batch matches world_offsets() exactly.
            n_envs_per_row=max(1, self.env_count),
            center_envs_at_origin=True,
        )
        self.built = True

        for handle in self.robot_handles:
            apply_drive_settings(
                handle, arm_kp, arm_kd, gripper_kp, gripper_kd, arm_armature, gripper_armature
            )

        self._home_dofs = self._home_dof_values()
        self._home_qs = self.read_piece_qs().copy()
        self.reset()
        return self.gs_scene

    @property
    def robot_handles(self) -> tuple[RobotHandle, ...]:
        """One handle per *distinct* arm entity (batched worlds share one)."""
        seen: dict[int, RobotHandle] = {}
        for world in self.worlds:
            seen.setdefault(id(world.robot.entity), world.robot)
        return tuple(seen.values())

    def _require_built(self) -> None:
        if not self.built:
            raise RuntimeError("ChessScene.finalize() has not been called yet")

    def _home_dof_values(self) -> np.ndarray:
        """``(env_count, len(watch_dofs))`` of the home posture with the gripper open."""
        values = np.zeros((self.env_count, len(self.watch_dofs)))
        for world in self.worlds:
            home, _ = world.robot.home_positions(opened=True)
            values[world.env, list(world.arm_dof_slots) + list(world.gripper_dof_slots)] = home
        return values

    @property
    def home_dofs(self) -> np.ndarray:
        """The home posture as a ``(env_count, len(watch_dofs))`` array. Read-only copy."""
        self._require_built()
        return self._home_dofs.copy()

    @property
    def spawn_piece_qs(self) -> np.ndarray:
        """Every piece's spawn pose, in the layout the solver stores it. Read-only copy."""
        self._require_built()
        return self._home_qs.copy()

    def apply_home_targets(self) -> None:
        """Command every arm to its home posture with the gripper open.

        Genesis' position control holds the last commanded target, and an unset target is
        zero -- which for most of these arms means folded flat through the table -- so this
        has to happen before the first step, not merely at reset.
        """
        self.write_dof_targets(self._home_dofs)

    def reset(self) -> None:
        """Put every joint and every piece back to its spawn state."""
        self._require_built()
        self.write_dof_positions(self._home_dofs)
        self.write_piece_qs(self._home_qs)
        self.apply_home_targets()
        self.zero_velocities()
        self.sim_time = 0.0
        self.frame_index = 0

    @property
    def renders(self) -> bool:
        """Whether anything in this scene will look at a rendered frame.

        Decided from the scene's own state -- does it have a camera, does it have a viewer --
        rather than passed in by the caller.  Every caller that had to answer this question
        itself got it wrong at least once: it is the same predicate as "should the frame loop
        install a capture hook" and "does --num-frames mean anything", and keeping one copy
        of it here is what stops those drifting apart.
        """
        return self.camera is not None or self.gs_scene.viewer is not None

    def step(self) -> None:
        """One rendered frame's worth of physics.

        ``update_visualizer`` is off unless something is going to look at the result:
        Genesis copies every link and vertex transform into the render context on each
        ``scene.step()`` even with no camera and no viewer, which on the 4x4 board is
        22.9 ms/frame against 73.0 with it -- a 3.2x tax for a picture nobody takes.  When
        it *is* on, only the last substep of a frame pays it, since the intermediate
        substeps are never displayed.
        """
        self._require_built()
        want = self.renders
        for index in range(self.sim_substeps):
            last = index == self.sim_substeps - 1
            self.gs_scene.step(update_visualizer=want and last, refresh_visualizer=want and last)
        self.frame_index += 1
        # Derived from the frame counter rather than accumulated, so a long run's frame
        # times stay exact.
        self.sim_time = self.frame_index * self.frame_dt

    ##
    # Batched state access
    ##

    def read_link_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """``(positions, xyzw quaternions)`` of every watched link.

        Both ``(env_count, len(watch_links), ...)``.  One kernel launch each rather than one
        per entity: a 32-piece board would otherwise be 33 launches per control tick.
        Quaternions are converted out of Genesis' wxyz here, which is the package's rule --
        see :mod:`robochess_genesis.gsmath`.
        """
        self._require_built()
        solver = self.gs_scene.rigid_solver
        positions = as_numpy(solver.get_links_pos(links_idx=self.watch_links))
        quats = from_gs_quat(solver.get_links_quat(links_idx=self.watch_links))
        return positions.reshape(self.env_count, -1, 3), quats.reshape(self.env_count, -1, 4)

    def read_link_velocities(self) -> np.ndarray:
        """Linear velocity of every watched link, ``(env_count, len(watch_links), 3)``.

        ``ref="link_com"`` rather than Genesis' ``"link_origin"`` default, to match what the
        Newton port's "is it at rest" test reads: ``newton.State.body_qd``'s first three
        entries are documented as "linear velocity relative to the body's center of mass in
        world frame".  The two differ by ``omega x r`` -- for a 50 mm piece that is only
        material while it is tumbling, but a tumbling piece is exactly what the
        ``still_moving`` clause exists to catch, so there is no reason to read a different
        point than the reference does.
        """
        self._require_built()
        velocities = as_numpy(
            self.gs_scene.rigid_solver.get_links_vel(links_idx=self.watch_links, ref="link_com")
        )
        return velocities.reshape(self.env_count, -1, 3)

    def read_dof_positions(self) -> np.ndarray:
        """``(env_count, len(watch_dofs))`` joint positions."""
        self._require_built()
        values = as_numpy(self.gs_scene.rigid_solver.get_dofs_position(dofs_idx=self.watch_dofs))
        return values.reshape(self.env_count, -1)

    def write_dof_targets(self, values: np.ndarray) -> None:
        """Command every watched DOF, ``(env_count, len(watch_dofs))``.

        Goes through the solver's global arrays rather than per-entity
        ``control_dofs_position`` so that one call covers every arm of every world in both
        multi-world modes.
        """
        self._require_built()
        self.gs_scene.rigid_solver.control_dofs_position(
            np.ascontiguousarray(values, dtype=np.float32), dofs_idx=self.watch_dofs
        )

    def write_dof_positions(self, values: np.ndarray) -> None:
        """Teleport every watched DOF, ``(env_count, len(watch_dofs))``."""
        self._require_built()
        self.gs_scene.rigid_solver.set_dofs_position(
            np.ascontiguousarray(values, dtype=np.float32), dofs_idx=self.watch_dofs
        )

    def read_piece_qs(self) -> np.ndarray:
        """Free-joint coordinates of every piece, ``(env_count, n_pieces * 7)``.

        Layout per piece is Genesis': ``x, y, z`` then a **wxyz** quaternion.  This is the
        one array in the package that carries wxyz, because it is a raw view of the solver's
        own storage, and the two call sites that touch it (reset randomisation) say so.
        """
        self._require_built()
        values = as_numpy(self.gs_scene.rigid_solver.get_qpos(qs_idx=self.watch_qs))
        return values.reshape(self.env_count, -1)

    def write_piece_qs(self, values: np.ndarray) -> None:
        """Teleport every piece, ``(env_count, n_pieces * 7)``; see :meth:`read_piece_qs`."""
        self._require_built()
        self.gs_scene.rigid_solver.set_qpos(
            np.ascontiguousarray(values, dtype=np.float32), qs_idx=self.watch_qs
        )

    def zero_velocities(self) -> None:
        """Stop everything: every DOF of every entity, in every env.

        Not just the watched DOFs -- the pieces' free joints are 6 DOFs each and are not in
        ``watch_dofs`` (a free joint's 7 coordinates and 6 DOFs do not line up, so the
        pieces are watched by *coordinate* instead).  Leaving them moving across a reset
        makes the first frames of an episode look like the board was shoved.
        """
        self._require_built()
        solver = self.gs_scene.rigid_solver
        solver.set_dofs_velocity(np.zeros((self.env_count, solver.n_dofs), dtype=np.float32))

    ##
    # Camera
    ##

    def default_camera(self, zoom: float = 1.0) -> tuple[Vec3, Vec3]:
        """A ``(eye, target)`` pair that frames every world's board and arm."""
        xs = [world.board_center[0] + offset[0] for world, offset in zip(self.worlds, self.render_offsets)]
        ys = [world.board_center[1] + offset[1] for world, offset in zip(self.worlds, self.render_offsets)]
        target = (
            (min(xs) + max(xs)) / 2.0,
            (min(ys) + max(ys)) / 2.0,
            self.table_top_z + CAMERA_TARGET_HEIGHT,
        )
        half_x = bl.TABLE_SIZE[0] / 2.0 + (max(xs) - min(xs)) / 2.0
        half_y = bl.TABLE_SIZE[1] / 2.0 + (max(ys) - min(ys)) / 2.0
        # A row of worlds along y lands across the frame's long axis, so it needs less
        # distance than the same extent along x -- but only sqrt(aspect) of it, because the
        # row also spreads in depth.
        radius = CAMERA_RADIUS_FACTOR * max(half_x, half_y / math.sqrt(CAMERA_ASPECT)) / max(zoom, 1e-3)

        # A three-quarter view reads best for one world, but with several worlds in a row
        # along y the near one stands in front of the far ones, so swing towards a frontal
        # view as the row grows.
        frontal = min(1.0, (max(ys) - min(ys)) / (2.0 * bl.TABLE_SIZE[1]))
        direction = np.array([-(0.55 + 0.40 * frontal), -0.66 * (1.0 - frontal), 0.55])
        direction /= np.linalg.norm(direction)
        eye = tuple(float(t + radius * d) for t, d in zip(target, direction))
        return eye, target

    def render(self) -> np.ndarray | None:
        """Render one RGB frame from the scene camera, or ``None`` if there is none."""
        if self.camera is None:
            return None
        return self.camera.render(rgb=True)[0]

    ##
    # Introspection
    ##

    def describe(self) -> str:
        """One line per world plus a totals line, for the scripts to print."""
        lines = []
        for world in self.worlds:
            center = world.board_center_world
            lines.append(
                f"  world {world.index}: {world.spec.key:<12s} {world.layout.scenario:<6s} "
                f"board={world.layout.board_usd:<20s} scale={world.layout.board_scale:.2f} "
                f"center=({center[0]:+.3f},{center[1]:+.3f}) pieces={world.piece_count:2d} "
                f"env={world.env} ee_link={world.robot.ee_link} "
                f"arm_dofs={world.robot.arm_dofs[0]}..{world.robot.arm_dofs[-1]} "
                f"reachable={len(world.reachable_piece_indices())}/{world.piece_count} "
                f"targets={len(world.target_positions())}"
            )
        if not self.homogeneous:
            lines.append(
                "  worlds differ, so they are laid side by side inside one Genesis env "
                "(a batched env is a copy of the scene, not a variant of it)"
            )
        totals = (
            f"  total: worlds={self.world_count} genesis_envs={self.env_count} "
            f"entities={len(self.gs_scene.entities)} "
            f"max_collision_pairs={self.contact_max} visual={self.visual}"
        )
        if self.built:
            solver = self.gs_scene.rigid_solver
            totals += (
                f" links={solver.n_links} geoms={solver.n_geoms} dofs={solver.n_dofs}"
                f" backend={_backend_name()}"
            )
        return "\n".join([*lines, totals])
