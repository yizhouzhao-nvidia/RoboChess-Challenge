"""Robot arms available to the Newton port of the RoboChess tasks.

The Isaac Lab task describes an arm with an :class:`ArticulationCfg` plus a
``ChessRobotSpec`` of hand-measured geometry (see
``lab/source/robochess/tasks/manager_based/chess/robot_configs.py``).  Newton has
no articulation config: an arm is a URDF/MJCF/USD import into a
:class:`newton.ModelBuilder`, and everything the scene and the pick controller
need afterwards -- which DOFs are the arm, which body is the hand, where the TCP
sits in that body -- has to be carried alongside the importer call.  That is what
:class:`NewtonRobotSpec` is.

The geometric fields (``tcp_offset``, ``approach_axis``, ``closing_axis``,
``reach``, ``board_distance``, ``home_joint_pos``) are carried over verbatim from
the Isaac Lab spec wherever the Newton source is the same physical asset, because
they were measured and the geometry does not change with the physics engine.  The
exceptions are called out in the per-spec comments: so101, ur10 and flexiv_rizon
have no Isaac Lab picking numbers at all, and the rebot and yam home postures had
to be re-solved because the Newton sources (newton-assets URDF, Menagerie MJCF)
use different joint frames from the USDs Isaac Lab loaded.

Asset sources are pinned by git ref so that the two supported interpreters
(newton 1.6 in ``~/Projects/newton``, newton 1.2.1 in the Isaac Lab venv) resolve
byte-identical files; their built-in ``NEWTON_ASSETS_REF`` / ``MENAGERIE_REF``
defaults differ, which would otherwise silently give the two runs different
inertias and joint limits.
"""

from __future__ import annotations

import os

# pxr's UsdPhysics.LoadUsdPhysicsFromRange() segfaults nondeterministically when it
# is allowed to use a thread pool, and ModelBuilder.add_usd() calls it. This has to
# be in place before the first pxr import, which happens lazily inside add_usd().
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

# board_layout owns both import-time guards (PXR_WORK_THREAD_LIMIT, and dropping the
# repo root from sys.path so this repo's newton/ directory cannot shadow the real
# package). Import it before newton, not for anything it exports.
from . import board_layout as _board_layout  # noqa: E402,F401

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton  # noqa: E402
import newton.utils  # noqa: E402
from newton import JointTargetMode  # noqa: E402

if Path(newton.__file__).resolve().is_relative_to(REPO_ROOT):
    raise ImportError(f"the repo directory shadows the newton package: newton.__file__ = {newton.__file__}")


__all__ = [
    "MENAGERIE_REF",
    "NEWTON_ASSETS_REF",
    "REBOT_SOURCE",
    "ROBOTS",
    "ROBOT_OPTIONS",
    "UNSUPPORTED_ROBOTS",
    "NewtonRobotSpec",
    "RobotHandle",
    "get_spec",
    "joint_target_positions",
    "joint_target_velocities",
    "load_robot",
    "menagerie_asset",
    "newton_asset",
]


def joint_target_positions(control: newton.Control) -> wp.array:
    """The position-target array of a :class:`newton.Control`, on either newton.

    1.5 renamed ``Control.joint_target_pos`` to ``joint_target_q`` and made the old
    name raise, so neither name works on both interpreters. Every arm here is a
    fixed-base chain of 1-DOF joints, so the DOF layout and the coordinate layout
    coincide and the array is indexed by the DOF indices on :class:`RobotHandle`
    either way.
    """
    return control.joint_target_q if hasattr(control, "joint_target_q") else control.joint_target_pos


def joint_target_velocities(control: newton.Control) -> wp.array:
    """The velocity-target array of a :class:`newton.Control`; see :func:`joint_target_positions`."""
    return control.joint_target_qd if hasattr(control, "joint_target_qd") else control.joint_target_vel


# newton-assets at this ref carries every folder we need -- franka_emika_panda,
# universal_robots_ur10 and seeed_rebot_devarm. The 1.2.1 default ref
# (8e8df07d) predates seeed_rebot_devarm entirely; the 1.6 default (a96f0973)
# has it but is a different snapshot of the other two.
NEWTON_ASSETS_REF = os.environ.get("ROBOCHESS_NEWTON_ASSETS_REF", "f8fb7abcbeba2318814a74f3eeb02780ad7925d6")

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
# newton 1.2.1 pins this ref; 1.6 pins da76818. Pin ours so both interpreters get
# the same yam finger limits and flexiv actuator gains.
MENAGERIE_REF = os.environ.get("ROBOCHESS_MENAGERIE_REF", "feadf76d42f8a2162426f7d226a3b539556b3bf5")


def newton_asset(folder: str) -> Path:
    """Path to a newton-assets folder at the ref this package is pinned to."""
    return newton.utils.download_asset(folder, ref=NEWTON_ASSETS_REF)


def menagerie_asset(folder: str) -> Path:
    """Path to a MuJoCo Menagerie folder at the ref this package is pinned to.

    ``download_git_folder`` lives in the private ``newton._src`` tree on both 1.2.1
    and 1.6 with an identical signature, but it is private, so a local checkout can
    be pointed at with ``ROBOCHESS_MENAGERIE_DIR`` if a future release moves it.
    """
    local = os.environ.get("ROBOCHESS_MENAGERIE_DIR")
    if local:
        return Path(local) / folder

    download = getattr(newton.utils, "download_git_folder", None)
    if download is None:
        from newton._src.utils.download_assets import download_git_folder as download

    return download(git_url=MENAGERIE_URL, folder_path=folder, ref=MENAGERIE_REF)


##
# Spec
##


@dataclass(frozen=True)
class NewtonRobotSpec:
    """Everything about an arm that the importer call itself does not record.

    ``arm_dofs`` / ``gripper_dofs`` are indices *relative to the first DOF this
    robot contributes to the builder*; :func:`load_robot` turns them into absolute
    model DOF indices. Body names are resolved after the import against
    ``builder.body_label``, because ``add_urdf`` and ``add_mjcf`` return ``None``
    and only ``add_usd`` hands back a prim-path map.
    """

    key: str

    load: Callable[[newton.ModelBuilder, wp.transform], object]
    """``(builder, xform) -> importer result``. Adds exactly one articulation."""

    source: str
    """Human-readable provenance, printed by the tools and quoted in bug reports."""

    arm_dofs: tuple[int, ...]
    gripper_dofs: tuple[int, ...]

    ee_body: str
    """Body the IK solver drives. ``approach_axis``/``closing_axis``/``tcp_offset``
    are all expressed in this body's frame."""

    finger_bodies: tuple[str, ...]

    tcp_offset: tuple[float, float, float]
    """Translation from :attr:`ee_body`'s origin to the point between the fingertips."""

    approach_axis: tuple[float, float, float]
    """Unit vector, in :attr:`ee_body` coordinates, pointing out of the gripper."""

    closing_axis: tuple[float, float, float]
    """Unit vector, in :attr:`ee_body` coordinates, along which the fingers travel."""

    gripper_open: tuple[float, ...]
    """Joint targets for :attr:`gripper_dofs`, in the same order, when open."""

    gripper_close: tuple[float, ...]

    max_opening: float
    """Fingertip separation at :attr:`gripper_open` [m]; must exceed the grasp span."""

    reach: float
    """Comfortable planar reach from the base [m]; bounds which squares are used."""

    board_distance: float
    """How far in front of the base the board centre is placed [m].

    Per-arm because these robots differ by a factor of five in reach: a board that
    is comfortable for a UR10 is entirely outside an SO-101's workspace.
    """

    base_pos: tuple[float, float, float]
    """Where the arm is bolted to the table, in world coordinates."""

    home_joint_pos: tuple[float, ...]
    """Reset posture, one value per entry of :attr:`arm_dofs`.

    Not cosmetic: several of these assets put the all-zeros pose flat through the
    table, which sweeps the board off before the first control step.
    """

    target_ke: float = 2000.0
    """Position-drive stiffness applied to every DOF of this arm.

    Most sources import with ``ke = 0`` (the URDFs and the piper/ur10 USDs carry no
    MuJoCo actuator, and the Menagerie ``<general>`` actuators are not converted),
    which leaves the arm limp. 2000/100 was measured to hold every entry of
    :data:`ROBOTS` -- the four supported arms and the three unsupported ones --
    within 0.026 rad of its home pose even with gravity compensation off. Picking
    needs a different split; see ``pick.apply_pick_gains``.

    Two sources do author gains, and the override is still applied to both rather
    than deferring to them, because it measures no worse and keeps one number
    across the table. The SO-101 MJCF ships ``kp = 998.22``. The reBot
    ``usd_structured`` asset ships 900/60 on joints 1-3, 120/10 on 4-6 and 5000/41.28
    on the fingers; holding its home pose for 120 frames, rebot/4x4, newton 1.6:

        gravity compensation on   authored 2.861e-08 rad   override 2.861e-08 rad
        gravity compensation off  authored 6.305e-03 rad   override 2.508e-03 rad

    i.e. identical where it matters and 2.5x tighter where it does not. Picking
    overwrites both anyway.
    """

    target_kd: float = 100.0

    gravity_compensation: bool = True
    """Set ``mujoco:jnt_actgravcomp`` on the arm DOFs and ``mujoco:gravcomp = 1``
    on every body of the arm, the way ``newton/examples/ik/example_ik_cube_stacking.py``
    does. This is what takes steady-state tracking from centimetres to ~1 mm, which
    is the difference between gripping a 19 mm bishop shaft and knocking it over."""

    @property
    def has_gripper(self) -> bool:
        return len(self.gripper_dofs) > 0

    @property
    def arm_dof_count(self) -> int:
        return len(self.arm_dofs)

    @property
    def gripper_dof_count(self) -> int:
        return len(self.gripper_dofs)

    def gripper_targets(self, opened: bool) -> tuple[float, ...]:
        return self.gripper_open if opened else self.gripper_close

    def graspgen_to_ee(self, graspgen_depth: float) -> np.ndarray:
        """4x4 transform taking a GraspGen grasp pose to this arm's ``ee_body`` pose.

        GraspGen's convention is approach ``+Z``, fingers closing along ``+X``,
        origin at the gripper base with the TCP at ``+Z * depth``. Real arms
        disagree on both axes -- the Franka closes along its hand's Y, the reBot
        reaches along its ``gripper_end``'s X -- so the mapping is built from the
        measured axes rather than assumed to be a rotation about one of them.

        Carried over unchanged from ``ChessRobotSpec.graspgen_to_ee``; the rows (not
        columns) matter, see that docstring.
        """
        approach = np.asarray(self.approach_axis, dtype=float)
        closing = np.asarray(self.closing_axis, dtype=float)
        approach /= np.linalg.norm(approach)
        closing /= np.linalg.norm(closing)
        rotation = np.array([closing, np.cross(approach, closing), approach])

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array([0.0, 0.0, graspgen_depth]) - rotation @ np.asarray(self.tcp_offset, dtype=float)
        return transform


##
# Loaders
##


def _load_franka(builder: newton.ModelBuilder, xform: wp.transform):
    # The Menagerie panda.xml has a <tendon><fixed> coupling the two finger joints
    # whose joint indices are not remapped when fixed joints collapse, which makes
    # SolverMuJoCo's tendon setup index out of bounds. The newton-assets FR3 URDF
    # has no tendon and carries a real fr3_hand_tcp body, so it is the better source
    # even though MJCF is normally preferred.
    return builder.add_urdf(
        str(newton_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf"),
        xform=xform,
        floating=False,
        # collapse would merge fr3_hand and fr3_hand_tcp into fr3_link7, i.e. destroy
        # the end-effector frame the IK drives.
        collapse_fixed_joints=False,
        enable_self_collisions=False,
    )


def _load_so101(builder: newton.ModelBuilder, xform: wp.transform):
    # All three repo-local SO-101 USDs raise a USD composition error under Newton
    # (a dangling reference to </visuals/gripper_frame_link> that Isaac Sim
    # tolerates and Newton hard-fails on), so the MJCF next to them is the source.
    return builder.add_mjcf(
        str(REPO_ROOT / "assets" / "so101" / "TheRobotStudio" / "so101_new_calib.xml"),
        xform=xform,
        floating=False,
        # only drops the static `base` wrapper body; the gripperframe site survives
        collapse_fixed_joints=True,
        enable_self_collisions=False,
    )


def _load_ur10(builder: newton.ModelBuilder, xform: wp.transform):
    # Same asset Isaac Lab's UR10_CFG uses. Menagerie only ships the ur10*e*.
    return builder.add_usd(
        str(newton_asset("universal_robots_ur10") / "usd" / "ur10_instanceable.usda"),
        xform=xform,
        collapse_fixed_joints=False,  # keeps ee_link as its own body
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )


def _load_piper(builder: newton.ModelBuilder, xform: wp.transform):
    # Repo-local USD rather than Menagerie agilex_piper/piper.xml: it is the asset
    # the Isaac Lab visual task spawns, so the two ports show the same robot, and
    # its fingers travel 50 mm each against the Menagerie model's 35 mm. The
    # Menagerie MJCF loads fine and is a drop-in alternative if a pure-MJCF scene
    # is ever wanted.
    return builder.add_usd(
        str(REPO_ROOT / "assets" / "piper" / "piper_camera.usd"),
        xform=xform,
        collapse_fixed_joints=False,  # keeps arm_base for mounting and camera_link
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )


REBOT_SOURCE = os.environ.get("ROBOCHESS_REBOT_SOURCE", "usd").strip().lower()
"""Which newton-assets ``seeed_rebot_devarm`` layer the reBot is imported from.

``"usd"`` (the default) is ``usd_structured/seeed_rebot_devarm.usda``, ``"urdf"`` is
``urdf/seeed_rebot_devarm.urdf``. Read once at import, so

    ROBOCHESS_REBOT_SOURCE=urdf python newton/scripts/run_chess_pick.py --robot rebot ...

A/Bs the two without editing anything, on either script. The URDF was the default
until the USD was measured against it and is kept reachable as the regression
baseline; it is not otherwise recommended. Pick success, ``--robot rebot
--world-count 8 --num-episodes 16 --seed 0 --viewer null`` (``8x8`` at 6/12), three
runs per cell, newton 1.6:

    scenario   usd_structured        urdf
    pieces     16/16 16/16 16/16     6/16  7/16  7/16
    1d         16/16 16/16 16/16     8/16  8/16  7/16
    3x3        16/16 16/16 16/16     16/16 15/16 15/16
    4x4        16/16 16/16 16/16     13/16 13/16 12/16
    8x8         5/12  4/12  5/12     3/12  3/12  2/12

and ``4x4`` on newton 1.2.1, where this arm was previously unusable:

    4x4        14/16 15/16 15/16     1/16 (README, unchanged)
"""

_REBOT_SOURCES = ("usd", "urdf")

if REBOT_SOURCE not in _REBOT_SOURCES:
    raise ValueError(f"ROBOCHESS_REBOT_SOURCE={REBOT_SOURCE!r}, expected one of {_REBOT_SOURCES}")


def _load_rebot_usd(builder: newton.ModelBuilder, xform: wp.transform):
    """The mujoco-usd-converter output of Menagerie's ``seeed_rebot_devarm.xml``.

    The win is the collision geometry, and it is a shape problem rather than a
    resolution one. The URDF's per-link convex hull is a *wedge*: measured as the free
    interval along the closing axis, it narrows from 83.5 mm at 5 mm depth to 20.0 mm
    at 64 mm, about 1.05 mm per mm. The USD's 8-12 part decomposition is a true
    parallel jaw -- 88.25 mm at every depth. (The "71 mm usable slot at the TCP" this
    port used to document was that wedge measured at ``tcp_offset``, an artefact of
    ``approximate_meshes("convex_hull")`` rather than a property of the arm.) Ablation:
    giving the URDF path the USD's damping and frictionloss is worth 1-2 episodes of
    the ~10-episode gain; the rest is the geometry, so it cannot be tuned for.

    Two things survive to ``mj_model`` that the override does not touch: joint
    ``damping`` (5/5/5/2/2/2/1/1) and ``frictionloss`` (0.2). ``armature`` is
    overridden. Cost: +16 % per step on 8x8, where 82 extra colliders per arm meet 32
    pieces; within noise on every smaller board.

    Same ten bodies, same eight joints, the same labels and the same masses as the
    URDF next to it, so every geometric field of the spec carries over unchanged --
    but the physics is authored rather than defaulted, and that is the reason to
    prefer it:

    * **Colliders.** 92 ``CONVEX_MESH`` parts (8-12 per link, 5,585 vertices, largest
      64) against the URDF's ten raw ``MESH`` colliders at 364,392 vertices. The
      URDF path has to run :meth:`~newton.ModelBuilder.approximate_meshes` to be
      steppable at all; calling it here would *lose* information -- one hull per link
      inflates collision volume to 3.24x the true link volume against 1.89x for the
      decomposition (asset README) -- as well as costing the hull solve, so it is
      not called.
    * **Drives.** ``ke = 900/900/900/120/120/120`` and ``kd = 60/60/60/10/10/10`` on
      the arm, 5000/41.28 on the fingers, plus ``armature = 0.01``, joint damping
      5/5/5/2/2/2/1/1 and ``frictionloss = 0.2``, all imported on both interpreters.
      The URDF imports ``ke = kd = armature = damping = 0``. :func:`load_robot` still
      overrides ``ke``/``kd``; see :attr:`NewtonRobotSpec.target_ke` for why.
    * **Finger coupling.** ``NewtonMimicAPI``/``MjcEqualityJointAPI`` 1:1 between
      ``joint_left`` and ``joint_right``. Honoured on newton 1.6 (one
      ``mujoco:equality_constraint`` row, joints 8 and 9, polycoef ``[0, 1, 0, 0, 0]``)
      and *dropped* on 1.2.1, whose importer looks for an ``mjc:target`` relationship
      the converter does not author and warns ``MjcEqualityJointAPI on
      '.../joint_left' has no mjc:target relationship; skipping``. Both fingers are
      commanded explicitly either way -- which is what the URDF already required --
      so the drop is survivable but not free. Worst ``|q_left - q_right|`` over four
      rebot/4x4 pick episodes, logged every control tick:

          USD, newton 1.6   (coupled)    2.1e-4 m, during ``transfer``
          USD, newton 1.2.1 (dropped)    2.6e-3 m, during ``close``
          URDF, newton 1.6  (no such constraint in the asset)  1.5e-3 m

      i.e. the coupling buys an order of magnitude in jaw symmetry where it is
      honoured, and where it is not the arm is no worse off than it was on the URDF.

    ``enable_self_collisions=False`` matches every other arm here, which also makes
    the asset's eleven ``physics:filteredPairs`` moot -- they exist to make
    self-collision usable, and nothing in this port turns it on.

    The one regression: the decomposition's parts bulge 1-2 mm past the concave
    surface (the asset README says so), and ``base_link``'s eight parts span
    ``x -0.2718..-0.1282, y +-0.1018, z 0.7682..0.8468`` in world coordinates against
    the hull's ``x -0.2700..-0.1300, y +-0.1000, z 0.7700..0.8450`` -- 1.8 mm proud on
    every face. ``8x8`` is the only board that stands a piece against that: white
    bishop 0, at ``(-0.0958, -0.0966)``, settles 1.94 mm *high* in three runs out of
    three (``dz_mm -1.71..+1.94``) where the hull leaves it at -0.4 mm. Every other
    piece and every other scenario settles in the usual -1.67..-1.25 band, and the
    ``8x8`` pick rate goes up rather than down (4-5/12 against 2-3/12), so this is
    logged, not fixed.
    """
    return builder.add_usd(
        str(newton_asset("seeed_rebot_devarm") / "usd_structured" / "seeed_rebot_devarm.usda"),
        xform=xform,
        collapse_fixed_joints=False,  # keeps gripper_end, which is the IK target
        enable_self_collisions=False,
        hide_collision_shapes=True,  # 92 collider parts on top of 23 render meshes
    )


def _load_rebot_urdf(builder: newton.ModelBuilder, xform: wp.transform):
    first_shape = builder.shape_count
    result = builder.add_urdf(
        str(newton_asset("seeed_rebot_devarm") / "urdf" / "seeed_rebot_devarm.urdf"),
        xform=xform,
        floating=False,
        collapse_fixed_joints=False,  # keeps gripper_end, which is the IK target
        enable_self_collisions=False,
    )

    # The reBot URDF points its <collision> tags at the same full-resolution STLs as
    # its <visual> tags, so the import lands ten raw GeoType.MESH colliders totalling
    # 364,392 vertices (largest 81,884) -- where the piper's USD gives CONVEX_MESH at
    # 640. Newton's narrow phase pays for that per pair, and the raw-mesh path is also
    # the numerically fragile one: with the untouched meshes this arm ran at 15.5
    # ms/step against 3.1 after this call (probe, newton 1.6, one arm on a table, home
    # pose, 120 steps at 1/240) and never completed a pick inside its episode budget.
    # Only the colliders are approximated, so the render meshes keep their detail; the
    # hulls are per link, and the palm's hull stops 58 mm short of the TCP, so it does
    # not bridge the jaw slot. Verified unchanged afterwards: max |q - target| over the
    # arm 1.03e-12 rad, and the jaws still close to a 0.00015 m pad gap.
    hull_shapes = [
        shape
        for shape in range(first_shape, builder.shape_count)
        if builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES
    ]
    builder.approximate_meshes(method="convex_hull", shape_indices=hull_shapes, raise_on_failure=True)
    return result


def _load_rebot(builder: newton.ModelBuilder, xform: wp.transform):
    if REBOT_SOURCE == "urdf":
        return _load_rebot_urdf(builder, xform)

    first_actuator = _actuator_row_count(builder)
    result = _load_rebot_usd(builder, xform)
    # The USD carries eight MjcActuator prims. BOTH importers build all eight rows
    # identically -- forcerange +-36 Nm on joints 1-3, +-14 on 4-6, +-1904 N on the
    # two fingers, ctrlrange equal to the joint limits. The versions diverge in the
    # SOLVER, not the importer: mj_model.actuator_forcerange keeps those values on
    # newton 1.6 and is (0, 0) on 1.2.1 whatever the builder rows say. Those limits size the
    # asset's own kp of 120-900; this port drives at 2000-4000 and routes gravity
    # compensation through the same actuators, so keeping them both clamps the drive
    # and makes the two interpreters disagree. Same call, same reasoning as yam -- see
    # _clear_imported_actuator_limits -- but here it is worth 9 of 16 episodes:
    # rebot/4x4, 8 worlds x 16 at seed 0, newton 1.6, three runs each,
    #     cleared  16/16 16/16 16/16      kept  7/16 7/16 7/16
    # and the kept runs fail as missed_target (8 of the 9 failures), which is the
    # drive saturating and the arm lagging its command: placement error over the
    # episodes that still succeed runs 3.6-12.0 mm against 1.1-2.6 mm cleared, and
    # the ones that do not land 67-340 mm from the destination square.
    _clear_imported_actuator_limits(builder, first_actuator)
    return result


def _load_yam(builder: newton.ModelBuilder, xform: wp.transform):
    first_actuator = _actuator_row_count(builder)
    result = builder.add_mjcf(
        str(menagerie_asset("i2rt_yam") / "yam.xml"),
        xform=xform,
        floating=False,
        collapse_fixed_joints=True,  # only the static `arm` wrapper body
        enable_self_collisions=False,
    )
    _clear_imported_actuator_limits(builder, first_actuator)
    return result


def _actuator_row_count(builder: newton.ModelBuilder) -> int:
    attribute = builder.custom_attributes.get("mujoco:actuator_forcerange")
    return 0 if attribute is None or attribute.values is None else len(attribute.values)


def _clear_imported_actuator_limits(builder: newton.ModelBuilder, first_row: int) -> None:
    """Drop the force and control clamps an MJCF import puts on its actuator rows.

    ``SolverMuJoCo`` builds one actuator per DOF and overwrites ``gainprm``/``biasprm``
    from :attr:`joint_target_ke`/:attr:`joint_target_kd` -- but on newton 1.6 it keeps
    the ``forcerange`` and ``ctrlrange`` the importer read out of the MJCF, and on
    1.2.1 it ignores them (every row comes out ``(0, 0)`` = unlimited). Measured on
    ``yam.xml``, which is the only one of the four arms whose asset carries actuators:

        newton 1.6    forcerange (+-28, +-28, +-28, +-10, +-10, +-10), forcelimited True
                      ctrlrange on the left finger (0, 0.041)
        newton 1.2.1  forcerange all (0, 0), forcelimited False, ctrlrange all (0, 0)

    Keeping the clamps is not the conservative choice, it is an incoherent one: they
    size the real arm's motors against the MJCF's own kp of 10-40, while everything
    here drives at 600-4000, so a 10 mrad tracking error already asks for more than
    the limit and the excess is thrown away -- including the gravity compensation,
    which ``mujoco:jnt_actgravcomp`` routes through these same actuators. The finger
    clamp is worse than a saturation: yam's closed command is -0.00205 m, which
    ``ctrlrange`` rounds up to 0, so the jaws stop 2 mm short of the grip.

    Clearing them makes newton 1.6 agree with 1.2.1 rather than the other way round.
    A no-op for arms whose source carries no actuator rows, but NOT for the rebot USD:
    that asset imports eight of them, and this call is what keeps them from clamping
    the drive. Keeping them costs 9 of 16 episodes on rebot/4x4 (7/16 against 16/16,
    three runs each, failing missed_target).
    """
    limits = {
        "mujoco:actuator_forcerange": wp.vec2f(0.0, 0.0),
        "mujoco:actuator_has_forcerange": 0,
        "mujoco:actuator_forcelimited": 0,
        "mujoco:actuator_ctrlrange": wp.vec2f(0.0, 0.0),
        "mujoco:actuator_has_ctrlrange": 0,
        "mujoco:actuator_ctrllimited": 0,
    }
    for key, cleared in limits.items():
        attribute = builder.custom_attributes.get(key)
        if attribute is None or attribute.values is None:
            continue
        for row in range(first_row, len(attribute.values)):
            attribute.values[row] = cleared


def _load_flexiv(builder: newton.ModelBuilder, xform: wp.transform):
    # Menagerie has the Rizon 4; Isaac Lab used the Rizon 4S. Neither has a gripper.
    return builder.add_mjcf(
        str(menagerie_asset("flexiv_rizon4") / "flexiv_rizon4.xml"),
        xform=xform,
        floating=False,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
    )


##
# The table
##


ROBOTS: dict[str, NewtonRobotSpec] = {
    "franka": NewtonRobotSpec(
        key="franka",
        load=_load_franka,
        source="newton-assets franka_emika_panda/urdf/fr3_franka_hand.urdf",
        arm_dofs=(0, 1, 2, 3, 4, 5, 6),  # fr3_joint1..7
        gripper_dofs=(7, 8),  # fr3_finger_joint1/2
        ee_body="fr3_hand",
        finger_bodies=("fr3_leftfinger", "fr3_rightfinger"),
        # Isaac Lab's panda_hand -> TCP offset; the URDF's own fr3_hand_tcp body sits
        # at 0.1033 along the same axis, so the two agree to 0.1 mm.
        tcp_offset=(0.0, 0.0, 0.1034),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(0.0, 1.0, 0.0),
        gripper_open=(0.04, 0.04),
        gripper_close=(0.0, 0.0),
        max_opening=0.08,
        reach=0.68,
        board_distance=0.45,
        base_pos=(-0.23, 0.0, 0.77),
        home_joint_pos=(0.0, -0.35, 0.0, -2.20, 0.0, 1.90, 0.785),
    ),
    "so101": NewtonRobotSpec(
        key="so101",
        load=_load_so101,
        source="repo assets/so101/TheRobotStudio/so101_new_calib.xml",
        arm_dofs=(0, 1, 2, 3, 4),  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
        gripper_dofs=(5,),  # `gripper` is REVOLUTE on this arm, range [-0.17453, 1.74533]
        ee_body="gripper",
        finger_bodies=("moving_jaw_so101_v1",),
        # The MJCF's own `gripperframe` site, read out of builder.shape_transform;
        # it sits 8 mm off the body axis because the fixed jaw is off-centre. The
        # approach is that offset normalised and snapped to the axis it is within 5
        # degrees of. The jaw is a hinge, not a slide, so the closing direction is
        # only constant to first order: the hinge axis is the gripper body's -Y, so
        # the tip sweeps the XZ plane and leaves the arm at (0.94, 0, -0.35) near the
        # TCP. Snapped to -X, which keeps it exactly orthogonal to the approach --
        # graspgen_to_ee needs an orthonormal frame.
        tcp_offset=(-0.0079, -0.0002, -0.0981),
        approach_axis=(0.0, 0.0, -1.0),
        closing_axis=(-1.0, 0.0, 0.0),
        # +q opens; 1.0 rad gives 97.8 mm of jaw-tip separation, 1.74533 (the limit)
        # gives 141.5 mm, which is wider than the arm can usefully control.
        gripper_open=(1.0,),
        gripper_close=(-0.17453,),
        max_opening=0.0978,
        # FK sweep over the joint-limit box: 0.477 m maximum planar TCP reach with
        # the TCP above the table. The four arms Isaac Lab measured sit at 0.56-0.72
        # of their sweep maximum and put the board at ~0.7 * reach, so scale the same
        # way rather than inventing a number.
        reach=0.28,
        board_distance=0.19,
        base_pos=(-0.10, 0.0, 0.782),
        home_joint_pos=(0.0, -0.6, 0.9, 0.9, 0.0),
        # The MJCF ships <position class="sts3215" kp="998.22" kv="2.731">, which is
        # already a usable position drive; the override only keeps the arm on the
        # same footing as every other entry in this table.
    ),
    "ur10": NewtonRobotSpec(
        key="ur10",
        load=_load_ur10,
        source="newton-assets universal_robots_ur10/usd/ur10_instanceable.usda",
        arm_dofs=(0, 1, 2, 3, 4, 5),
        gripper_dofs=(),  # no hand in this asset -- visualization only
        ee_body="ee_link",
        finger_bodies=(),
        tcp_offset=(0.0, 0.0, 0.0),
        approach_axis=(1.0, 0.0, 0.0),  # ee_link's +X points out of the tool flange
        closing_axis=(0.0, 1.0, 0.0),
        gripper_open=(),
        gripper_close=(),
        max_opening=0.0,
        # FK sweep maximum planar reach above the table: 1.327 m.
        reach=0.80,
        board_distance=0.55,
        base_pos=(-0.42, 0.0, 0.77),
        home_joint_pos=(0.0, -1.2, 1.6, -1.95, -1.57, 0.0),
    ),
    "piper": NewtonRobotSpec(
        key="piper",
        load=_load_piper,
        source="repo assets/piper/piper_camera.usd",
        arm_dofs=(0, 1, 2, 3, 4, 5),  # joint1..6
        gripper_dofs=(6, 7),  # joint7 [0, 0.05], joint8 [-0.05, 0]
        ee_body="link6",
        finger_bodies=("link7", "link8"),
        tcp_offset=(0.0, 0.0, 0.125),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(1.0, 0.0, 0.0),
        # This USD lets the fingers travel 50 mm each (99.8 mm of measured pad
        # separation), where the asset the Isaac Lab picking task used stops at 35 mm.
        # Command the smaller travel: the pads track the joint exactly (measured pad
        # separation = 2 * q), so 0.035 reproduces the real Piper's 70 mm span.
        gripper_open=(0.035, -0.035),
        gripper_close=(0.0, 0.0),
        max_opening=0.07,
        reach=0.42,
        board_distance=0.30,
        base_pos=(-0.20, 0.0, 0.77),
        home_joint_pos=(0.0, 1.25, -1.55, 0.0, 0.95, 0.0),
    ),
    "rebot": NewtonRobotSpec(
        key="rebot",
        load=_load_rebot,
        source=(
            "newton-assets seeed_rebot_devarm/usd_structured/seeed_rebot_devarm.usda"
            if REBOT_SOURCE == "usd"
            else "newton-assets seeed_rebot_devarm/urdf/seeed_rebot_devarm.urdf"
        ),
        arm_dofs=(0, 1, 2, 3, 4, 5),  # joint1..6
        gripper_dofs=(6, 7),  # joint_left, joint_right, both [0, 0.05], same sign
        ee_body="gripper_end",
        finger_bodies=("gripper_left", "gripper_right"),
        tcp_offset=(-0.015, 0.0, 0.0),
        approach_axis=(1.0, 0.0, 0.0),
        closing_axis=(0.0, 1.0, 0.0),
        # Both fingers are always commanded. The URDF authors no mimic constraint at
        # all (grep -ci mimic on it is 0), so there is nothing to drop there; the USD
        # authors one, which newton 1.6 honours as an equality row and 1.2.1 skips
        # for want of an mjc:target relationship (see _load_rebot_usd). So no version
        # of this arm can be driven through joint_left alone. Worst measured jaw
        # drift over four full pick episodes: 2.1e-4 m coupled (USD on 1.6),
        # 2.6e-3 m uncoupled (USD on 1.2.1), 1.5e-3 m (URDF, no constraint). Measured on both sources and both interpreters:
        # the open command gives 0.09015 m of gripper_left/gripper_right body
        # separation and the closed command 0.00015 m.
        gripper_open=(0.045, 0.045),
        gripper_close=(0.0, 0.0),
        max_opening=0.09,
        reach=0.44,
        board_distance=0.30,
        base_pos=(-0.20, 0.0, 0.77),
        # Same for both sources: the USD's joint frames are the URDF's. Held for 120
        # frames at rest, the TCP lands on (0.09395, 0, 1.00961) either way, so
        # nothing below this line changed when the source did. Retuned for these
        # frames rather than Isaac Lab's (0, -1.25, -1.55, 0, -0.75, 0), which was
        # measured on the Seeed USD Isaac Lab spawns -- a different asset from the
        # newton-assets one -- and leaves the gripper pointing up and sideways. This
        # puts the TCP 0.24 m over the board centre with the approach axis 18 degrees
        # off straight down.
        home_joint_pos=(0.0, -1.50, -1.72, 1.48, 0.0, 0.0),
    ),
    "yam": NewtonRobotSpec(
        key="yam",
        load=_load_yam,
        source="mujoco_menagerie i2rt_yam/yam.xml",
        arm_dofs=(0, 1, 2, 3, 4, 5),  # joint1..6
        gripper_dofs=(6, 7),  # left_finger [-0.00205, 0.037524], right_finger mirrored
        ee_body="link_6",
        finger_bodies=("link_left_finger", "link_right_finger"),
        # NOT the Menagerie model's own `grasp_site` at (0, 0, 0.1347): that sits on
        # link_6's axis and the jaws do not. Measured on the finalized model, the 13
        # collision shapes of the two fingers span x = +-0.0379..0.0575, y = -0.049..0,
        # z = 0.0725..0.1385 in link_6's frame, mean (+-0.0441, -0.0367, 0.1106) -- the
        # slot between the jaws is centred 33 mm off the axis along -Y. A grasp aimed at
        # the axis closes on empty air 33 mm from the piece: measured, yam/4x4, the
        # descend converges to 2.9 mm, the jaws run down to a 0.03 opening fraction and
        # the piece never leaves the board, 0/8 episodes. Only the lateral component is
        # corrected; the depth keeps Isaac Lab's empirical 0.13 (it bracketed yam.usd at
        # 0.09 and 0.145, both 0 successes, against 12% at 0.13).
        tcp_offset=(0.0, -0.033, 0.13),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(1.0, 0.0, 0.0),
        # MIRRORED relative to the Isaac Lab yam.usd: here the joint extremes are
        # open and (-0.00205, +0.00205) is closed, the reverse of the lab's spec.
        gripper_open=(0.037524, -0.037524),
        gripper_close=(-0.00205, 0.00205),
        # Each pad travels 39.6 mm and the pads meet at close (7 mm of residual
        # centroid separation over 15 collision shapes), so the open span is the
        # 79 mm of total travel. Note the finger *body* origins move the opposite
        # way to the pads on this model -- measuring them gives the wrong sign.
        max_opening=0.079,
        reach=0.42,
        board_distance=0.30,
        base_pos=(-0.20, 0.0, 0.77),
        # Retuned for this asset: Isaac Lab's (0, 1.25, 1.25, 0, 0.85, 0) was measured
        # on yam.usd, whose joint frames differ from the Menagerie model's, and leaves
        # the tool pointing horizontally. This puts the TCP 0.24 m over the board
        # centre with the approach axis 16 degrees off straight down.
        home_joint_pos=(0.0, 1.49, 1.67, -1.48, 0.0, 0.0),
    ),
    "flexiv_rizon": NewtonRobotSpec(
        key="flexiv_rizon",
        load=_load_flexiv,
        source="mujoco_menagerie flexiv_rizon4/flexiv_rizon4.xml",
        arm_dofs=(0, 1, 2, 3, 4, 5, 6),  # joint1..7
        gripper_dofs=(),  # no hand in this asset -- visualization only
        ee_body="link7",
        finger_bodies=(),
        tcp_offset=(0.0, 0.0, 0.0),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(1.0, 0.0, 0.0),
        gripper_open=(),
        gripper_close=(),
        max_opening=0.0,
        # FK sweep maximum planar reach above the table: 0.896 m.
        reach=0.54,
        board_distance=0.38,
        base_pos=(-0.42, 0.0, 0.77),
        home_joint_pos=(0.0, -0.7, 0.0, 1.6, 0.0, 0.7, 0.0),
    ),
}

ROBOT_OPTIONS = ("franka", "piper", "rebot", "yam")
"""Arms this port supports, matching the Isaac Lab picking task's ``CHESS_ROBOTS``.

All four have a parallel-jaw gripper, which is what the GraspGen grasps assume. The
other three entries in :data:`ROBOTS` load and simulate correctly but are not part of
the supported set: ``ur10`` and ``flexiv_rizon`` ship no gripper at all, and the
``so101`` MJCF has a single moving jaw rather than an opposed pair, so a GraspGen
pinch does not retarget onto it. They stay reachable through :func:`get_spec` for
scene inspection; nothing in this package is verified against them.
"""

UNSUPPORTED_ROBOTS = tuple(key for key in ROBOTS if key not in ROBOT_OPTIONS)


def get_spec(robot: str | NewtonRobotSpec) -> NewtonRobotSpec:
    if isinstance(robot, NewtonRobotSpec):
        return robot
    try:
        return ROBOTS[robot]
    except KeyError as error:
        raise ValueError(f"Unsupported robot {robot!r}. Choose one of {ROBOT_OPTIONS}.") from error


##
# Loading
##


@dataclass(frozen=True)
class RobotHandle:
    """Where a loaded arm ended up in the builder, in absolute model indices.

    Builder indices survive :meth:`newton.ModelBuilder.finalize` unchanged, so the
    body and DOF indices here address the finalized :class:`newton.Model` directly.
    They do *not* survive being merged into another builder -- use :meth:`shifted`
    if the arm was built in a sub-builder that is later added to a scene builder.
    """

    spec: NewtonRobotSpec
    body_start: int
    body_count: int
    joint_start: int
    joint_count: int
    dof_start: int
    dof_count: int
    arm_dofs: tuple[int, ...]
    gripper_dofs: tuple[int, ...]
    ee_body: int
    finger_bodies: tuple[int, ...]
    body_labels: tuple[str, ...]
    """Labels of this robot's bodies, index i corresponding to body ``body_start + i``."""

    @property
    def body_stop(self) -> int:
        return self.body_start + self.body_count

    def body_index(self, name: str) -> int:
        """Absolute body index for a full label or a unique final path component."""
        return _resolve_body(self.body_labels, name, self.body_start)

    def shifted(self, delta_body: int = 0, delta_joint: int = 0, delta_dof: int = 0) -> RobotHandle:
        return replace(
            self,
            body_start=self.body_start + delta_body,
            joint_start=self.joint_start + delta_joint,
            dof_start=self.dof_start + delta_dof,
            arm_dofs=tuple(d + delta_dof for d in self.arm_dofs),
            gripper_dofs=tuple(d + delta_dof for d in self.gripper_dofs),
            ee_body=self.ee_body + delta_body,
            finger_bodies=tuple(b + delta_body for b in self.finger_bodies),
        )

    def write_joint_positions(self, joint_q: np.ndarray, opened: bool = True) -> np.ndarray:
        """Write the home pose (and the requested gripper state) into ``joint_q``.

        ``joint_q`` is a full-model DOF vector; it is modified in place and returned
        so this can be chained into ``model.joint_q.assign(...)``.
        """
        joint_q[list(self.arm_dofs)] = self.spec.home_joint_pos
        if self.gripper_dofs:
            joint_q[list(self.gripper_dofs)] = self.spec.gripper_targets(opened)
        return joint_q

    def write_gripper(self, joint_target: np.ndarray, opened: bool) -> np.ndarray:
        if self.gripper_dofs:
            joint_target[list(self.gripper_dofs)] = self.spec.gripper_targets(opened)
        return joint_target

    def tcp_pose(self, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World TCP pose from a ``state.body_q.numpy()`` array. Quaternion is xyzw."""
        pose = np.asarray(body_q[self.ee_body], dtype=float)
        pos, quat = pose[:3], pose[3:7]
        return pos + _quat_rotate(quat, np.asarray(self.spec.tcp_offset, dtype=float)), quat

    def ee_axes(self, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World approach and closing directions of the gripper."""
        quat = np.asarray(body_q[self.ee_body], dtype=float)[3:7]
        return (
            _quat_rotate(quat, np.asarray(self.spec.approach_axis, dtype=float)),
            _quat_rotate(quat, np.asarray(self.spec.closing_axis, dtype=float)),
        )


def load_robot(
    builder: newton.ModelBuilder,
    robot: str | NewtonRobotSpec,
    xform: wp.transform | None = None,
) -> RobotHandle:
    """Import an arm into ``builder`` and return where it landed.

    ``xform`` defaults to the spec's ``base_pos`` with identity rotation, which is
    where the RoboChess table expects it. ``SolverMuJoCo.register_custom_attributes``
    is called here as well as (harmlessly) by the caller: it is idempotent and does
    not clear attribute values that are already set, but it must run before the
    *first* ``add_*`` on the builder for the MuJoCo schema attributes of USD assets
    to be picked up, so a scene builder should still call it up front.
    """
    spec = get_spec(robot)
    if xform is None:
        xform = wp.transform(wp.vec3(*spec.base_pos), wp.quat_identity())

    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    body_start = builder.body_count
    joint_start = builder.joint_count
    dof_start = builder.joint_dof_count

    spec.load(builder, xform)

    body_count = builder.body_count - body_start
    joint_count = builder.joint_count - joint_start
    dof_count = builder.joint_dof_count - dof_start
    if dof_count == 0:
        raise RuntimeError(f"{spec.key}: the importer added no DOFs")

    expected = spec.arm_dof_count + spec.gripper_dof_count
    if dof_count != expected:
        raise RuntimeError(
            f"{spec.key}: expected {expected} DOFs from {spec.source}, got {dof_count}. "
            f"joint labels: {builder.joint_label[joint_start:]}"
        )

    arm_dofs = tuple(dof_start + d for d in spec.arm_dofs)
    gripper_dofs = tuple(dof_start + d for d in spec.gripper_dofs)

    for dof in range(dof_start, dof_start + dof_count):
        builder.joint_target_ke[dof] = spec.target_ke
        builder.joint_target_kd[dof] = spec.target_kd
        builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)

    if spec.gravity_compensation:
        _apply_gravity_compensation(builder, range(body_start, builder.body_count), arm_dofs)

    body_labels = tuple(builder.body_label[body_start:])
    return RobotHandle(
        spec=spec,
        body_start=body_start,
        body_count=body_count,
        joint_start=joint_start,
        joint_count=joint_count,
        dof_start=dof_start,
        dof_count=dof_count,
        arm_dofs=arm_dofs,
        gripper_dofs=gripper_dofs,
        ee_body=_resolve_body(body_labels, spec.ee_body, body_start),
        finger_bodies=tuple(_resolve_body(body_labels, name, body_start) for name in spec.finger_bodies),
        body_labels=body_labels,
    )


def _apply_gravity_compensation(
    builder: newton.ModelBuilder,
    bodies: Sequence[int] | range,
    arm_dofs: Sequence[int],
) -> None:
    """Cancel the arm's own weight inside MuJoCo, as example_ik_cube_stacking does.

    ``jnt_actgravcomp`` routes the compensation through the actuators (so it shows up
    as commanded torque rather than a free lunch) and is only meaningful on actuated
    arm DOFs; ``gravcomp`` is a per-body scale and is set on every link of the arm,
    including the hand and fingers. Bodies welded to the world have no DOFs, so
    setting it on them is a no-op.
    """
    joint_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
    if joint_attr.values is None:
        joint_attr.values = {}
    for dof in arm_dofs:
        joint_attr.values[dof] = True

    body_attr = builder.custom_attributes["mujoco:gravcomp"]
    if body_attr.values is None:
        body_attr.values = {}
    for body in bodies:
        body_attr.values[body] = 1.0


def _resolve_body(labels: Sequence[str], name: str, offset: int) -> int:
    """Body index for a full label or for a unique final path component.

    Labels are ``/``-separated and differ per importer: the FR3 URDF gives
    ``fr3/fr3_hand``, the UR10 USD ``/ur10/ee_link``, the SO-101 MJCF the whole
    ``worldbody/base/.../gripper`` chain. Matching the last component keeps the
    specs readable without hard-coding one importer's naming.
    """
    exact = [i for i, label in enumerate(labels) if label == name]
    if len(exact) == 1:
        return offset + exact[0]

    tail = [i for i, label in enumerate(labels) if label.rsplit("/", 1)[-1] == name]
    if len(tail) == 1:
        return offset + tail[0]
    # Both counts: a name that matches two labels *exactly* leaves `tail` empty, so
    # reporting only that would say "matched 0", i.e. the opposite of what happened.
    raise KeyError(
        f"body {name!r} matched {len(exact)} labels exactly and {len(tail)} by final "
        f"component, need exactly 1 of either, in {list(labels)}"
    )


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by an xyzw quaternion (warp's and Isaac Lab 3.0's convention)."""
    axis, w = quat[:3], quat[3]
    return vec + 2.0 * np.cross(axis, np.cross(axis, vec) + w * vec)
