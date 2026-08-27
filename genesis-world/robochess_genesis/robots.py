"""Robot arms available to the Genesis port of the RoboChess tasks.

Genesis, like Newton, has no articulation config: an arm is a URDF/MJCF/USD morph handed
to ``scene.add_entity``, and everything the scene and the pick controller need afterwards
-- which DOFs are the arm, which link is the hand, where the TCP sits in that link -- has
to be carried alongside the morph.  That is what :class:`GenesisRobotSpec` is, and it is a
field-for-field translation of ``robochess_newton.robots.NewtonRobotSpec``.

**The assets are the same files the Newton port loads**, at the same pinned git refs, so
the two ports show the same robot down to the mesh: the Franka and the reBot from
newton-assets, the YAM from MuJoCo Menagerie, the Piper from ``assets/piper`` in this repo.
The geometric fields (``tcp_offset``, ``approach_axis``, ``closing_axis``, ``reach``,
``board_distance``, ``home_joint_pos``) are therefore carried over verbatim -- they were
measured on these exact assets and do not change with the physics engine.  Every one of
them was re-verified against the Genesis import: same joint order, same limits, same link
names, which is what the DOF-count and limit checks in :func:`load_robot` enforce at load
time rather than trusting.

Three things *are* Genesis-specific, and each of them is a silent wrong answer if skipped:

* **``package://`` does not resolve.**  Genesis' primary URDF parser turns
  ``package://franka_emika_panda/meshes/x.stl`` into ``<urdf dir>/franka_emika_panda/
  meshes/x.stl``, which is one directory too deep for both of the newton-assets URDFs.
  The parse then fails, Genesis falls back to its legacy urdfpy parser, and *that* one
  raises a bare ``TypeError: Cannot cast array data from dtype('O')`` out of
  ``_init_dof_fields`` -- a stack trace with nothing in it about mesh paths.
  :func:`stage_urdf` rewrites the references to absolute paths in a cached copy.
* **URDF fixed links are merged by default.**  ``gs.morphs.URDF.merge_fixed_links`` is
  ``True``, which folds ``fr3_hand``/``fr3_hand_tcp`` into ``fr3_link7`` and ``gripper_end``
  into ``link6`` -- i.e. destroys the frame the IK drives -- so the two URDF arms name their
  links in ``links_to_keep``.  ``links_to_keep`` is a **URDF-only** field: passing it to
  ``gs.morphs.MJCF`` or ``gs.morphs.USD`` raises ``Unrecognized attribute``, and neither
  needs it (MuJoCo welds jointless bodies rather than fixed-jointing them, and the USD
  importer keeps every rigid-body prim).  :func:`load_robot` checks that every link a spec
  names actually survived, whichever importer ran.
* **Imported actuator gains and force ranges are honoured.**  The four assets import with
  wildly different drives (Franka kp=100/kv=40, YAM kp=40..0, Piper USD kp=0.19..59) and
  with force ranges sized against those gains.  :func:`load_robot` overwrites the gains and
  clears the ranges, for the same reason the Newton port clears the MuJoCo ones: a drive
  running at kp=4000 asks for more than the asset's motor limit after 10 mrad of error, and
  everything above the clamp -- including the gravity compensation -- is thrown away.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# board_layout first: it installs PXR_WORK_THREAD_LIMIT before anything can import pxr,
# which the Piper USD does. Not imported for what it exports.
from . import board_layout as _board_layout  # noqa: F401 -- imported for its side effect
from .assets import cache_root

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

__all__ = [
    "MENAGERIE_REF",
    "NEWTON_ASSETS_REF",
    "ROBOTS",
    "ROBOT_OPTIONS",
    "UNSUPPORTED_ROBOTS",
    "GenesisRobotSpec",
    "RobotHandle",
    "get_spec",
    "load_robot",
    "menagerie_asset",
    "newton_asset",
    "resolve_link",
    "stage_urdf",
]


##
# Asset acquisition
#
# Pinned by git ref so this port and the Newton one resolve byte-identical files. The
# refs are the Newton port's, verbatim.
##

NEWTON_ASSETS_URL = "https://github.com/newton-physics/newton-assets.git"
NEWTON_ASSETS_REF = os.environ.get("ROBOCHESS_NEWTON_ASSETS_REF", "f8fb7abcbeba2318814a74f3eeb02780ad7925d6")

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_REF = os.environ.get("ROBOCHESS_MENAGERIE_REF", "feadf76d42f8a2162426f7d226a3b539556b3bf5")


def _newton_cache_hit(prefix: str, folder: str, ref: str) -> Path | None:
    """An existing ``~/.cache/newton`` checkout of the same folder at the same ref.

    ``newton.utils.download_asset`` names its directories
    ``<repo>_<folder>_<hash>_<ref[:8]>``, where the hash is internal to newton.  Matching
    on the two parts that are meaningful reuses a checkout the Newton port already paid
    for -- 93 MB for the Franka, 86 MB for the reBot -- instead of cloning it twice on a
    machine that runs both ports.  Nothing is written there; a miss just falls through to
    this package's own cache.
    """
    cache = Path.home() / ".cache" / "newton"
    if not cache.is_dir():
        return None
    for entry in sorted(cache.glob(f"{prefix}_{folder}_*_{ref[:8]}")):
        candidate = entry / folder
        if candidate.is_dir():
            return candidate
    return None


def _download_git_folder(url: str, folder: str, ref: str, prefix: str) -> Path:
    """Sparse-checkout one folder of a git repo at a ref, into this package's cache.

    Blobless (``--filter=blob:none``) plus a sparse checkout so the Menagerie's 2 GB of
    other robots never lands on disk.  The clone is done into a temporary directory and
    renamed into place, so an interrupted download cannot leave a half-populated folder
    that the next run mistakes for a hit.
    """
    hit = _newton_cache_hit(prefix, folder, ref)
    if hit is not None:
        return hit

    root = cache_root() / "assets"
    target = root / f"{prefix}_{folder}_{ref[:8]}"
    if (target / folder).is_dir():
        return target / folder

    if shutil.which("git") is None:
        raise RuntimeError(
            f"git is needed to fetch {folder} from {url} and is not on PATH. Either install it, "
            f"or point this package at an existing checkout (ROBOCHESS_MENAGERIE_DIR for the "
            f"Menagerie)."
        )

    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{target.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        run = lambda *args: subprocess.run(args, cwd=staging, check=True, capture_output=True)  # noqa: E731
        run("git", "init", "-q")
        run("git", "remote", "add", "origin", url)
        run("git", "config", "core.sparseCheckout", "true")
        run("git", "sparse-checkout", "set", "--no-cone", folder)
        run("git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", ref)
        run("git", "checkout", "-q", "FETCH_HEAD")
    except subprocess.CalledProcessError as error:
        shutil.rmtree(staging, ignore_errors=True)
        stderr = error.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"failed to fetch {folder} from {url} at {ref}:\n{stderr}") from error
    if not (staging / folder).is_dir():
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"{url} at ref {ref} has no folder {folder!r}")
    staging.rename(target)
    return target / folder


def newton_asset(folder: str) -> Path:
    """Path to a newton-assets folder at the ref this package is pinned to."""
    return _download_git_folder(NEWTON_ASSETS_URL, folder, NEWTON_ASSETS_REF, "newton-assets")


def menagerie_asset(folder: str) -> Path:
    """Path to a MuJoCo Menagerie folder at the ref this package is pinned to."""
    local = os.environ.get("ROBOCHESS_MENAGERIE_DIR")
    if local:
        return Path(local) / folder
    return _download_git_folder(MENAGERIE_URL, folder, MENAGERIE_REF, "mujoco_menagerie")


def stage_urdf(urdf: Path, package_root: Path) -> Path:
    """A cached copy of *urdf* whose ``package://`` mesh references are absolute paths.

    Genesis resolves ``package://<pkg>/<rest>`` by dropping the scheme and joining the
    remainder onto the URDF's own directory, so ``<root>/franka_emika_panda/urdf/
    fr3_franka_hand.urdf`` looks for ``<root>/franka_emika_panda/urdf/franka_emika_panda/
    meshes/...`` -- one level too deep.  The correct package root is *``package_root``*,
    and substituting it in is both the smallest fix and the one that leaves the original
    checkout untouched (it may be a shared ``~/.cache/newton`` directory that the Newton
    port is also reading).

    Absolute paths are the substitution rather than a relative one because both of
    Genesis' URDF parsers ``os.path.join`` the reference onto the URDF directory, and
    joining an absolute path is the identity -- so this works whichever parser runs.

    Returns *urdf* unchanged if it has no ``package://`` reference.
    """
    text = urdf.read_text()
    if "package://" not in text:
        return urdf

    root = str(package_root.resolve()).rstrip("/") + "/"
    rewritten = text.replace("package://", root)
    target = cache_root() / "robots" / urdf.parent.parent.name / urdf.name
    target.parent.mkdir(parents=True, exist_ok=True)
    # Rewritten every time rather than cached on a stamp: it is a few hundred kB of string
    # replace against a checkout that is pinned by ref, so there is nothing to gain and a
    # stale copy pointing at a deleted cache directory to lose.
    if not target.exists() or target.read_text() != rewritten:
        target.write_text(rewritten)
    return target


##
# Spec
##


@dataclass(frozen=True)
class GenesisRobotSpec:
    """Everything about an arm that the morph itself does not record.

    ``arm_dofs`` / ``gripper_dofs`` are indices *local to this entity*, which is the
    indexing every ``RigidEntity`` method takes (``dofs_idx_local=``).  Link names are
    resolved after the entity is created, against ``entity.links``, because the four
    importers name links differently -- the USD one prefixes the whole prim path.
    """

    key: str

    morph: Callable[[Sequence[float]], Any]
    """``(base_position) -> gs.morphs.Morph``. Built lazily so importing this module does
    not import Genesis, and so a missing asset is an error at scene construction rather
    than at import."""

    source: str
    """Human-readable provenance, printed by the tools and quoted in bug reports."""

    arm_dofs: tuple[int, ...]
    gripper_dofs: tuple[int, ...]

    ee_link: str
    """Link the IK solver drives. ``approach_axis``/``closing_axis``/``tcp_offset`` are all
    expressed in this link's frame."""

    finger_links: tuple[str, ...]

    keep_links: tuple[str, ...] = ()
    """Links that must survive ``merge_fixed_links``. Always includes :attr:`ee_link` and
    :attr:`finger_links` -- listed separately only for links needed for some other reason."""

    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation from :attr:`ee_link`'s origin to the point between the fingertips."""

    approach_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Unit vector, in :attr:`ee_link` coordinates, pointing out of the gripper."""

    closing_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    """Unit vector, in :attr:`ee_link` coordinates, along which the fingers travel."""

    gripper_open: tuple[float, ...] = ()
    """Joint targets for :attr:`gripper_dofs`, in the same order, when open."""

    gripper_close: tuple[float, ...] = ()

    max_opening: float = 0.0
    """Fingertip separation at :attr:`gripper_open` [m]; must exceed the grasp span."""

    reach: float = 0.0
    """Comfortable planar reach from the base [m]; bounds which squares are used."""

    board_distance: float = 0.0
    """How far in front of the base the board centre is placed [m].

    Per-arm because these robots differ by a factor of five in reach: a board that is
    comfortable for a UR10 is entirely outside an SO-101's workspace."""

    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Where the arm is bolted to the table, in world coordinates."""

    home_joint_pos: tuple[float, ...] = ()
    """Reset posture, one value per entry of :attr:`arm_dofs`.

    Not cosmetic: several of these assets put the all-zeros pose flat through the table,
    which sweeps the board off before the first control step."""

    target_kp: float = 2000.0
    """Position-drive stiffness applied to every DOF of this arm.

    The four assets import with drives that have nothing in common -- the Franka URDF gets
    Genesis' 100/40 default, the YAM MJCF its authored 40/2.5 falling to 0 on the mimicked
    finger, the Piper USD a derived 0.19-59 -- so a single value here is what makes the
    four arms behave comparably.  2000/100 holds every supported arm within a few
    milliradians of its home pose with gravity compensation on, which is what visual
    inspection needs.  Picking needs a different split; see ``pick.apply_pick_gains``."""

    target_kd: float = 100.0

    armature: float = 0.1
    """Rotor inertia added to every DOF [kg m^2].

    Genesis' own URDF/MJCF default is 0.1 (``gs.morphs.URDF.default_armature``), and
    unlike Newton -- whose importers leave it at 0, which is why the Newton port had to
    discover armature the hard way -- that default is already in the range that makes a
    position-driven contact stable.  It is stated here rather than inherited so that all
    four arms agree whatever their asset says (the YAM MJCF authors 0.032/0.0018)."""

    gravity_compensation: float = 1.0
    """``gs.materials.Rigid(gravity_compensation=)`` for this arm, 1.0 = fully cancelled.

    Genesis' position control is a plain PD servo: force = ``kp * (target - q) - kv * qd``,
    with no feed-forward. Left to sag under its own weight, an arm at kp=2000 sits
    millimetres below its command, which is the difference between gripping a 19 mm bishop
    shaft and knocking it over."""

    pick_gripper_kp: float | None = None
    """Per-arm override of ``pick.PICK_GRIPPER_KP``; ``None`` uses the shared default.

    The four grippers are not the same mechanism -- their jaw spans run 70-90 mm, and the
    YAM's pads move through a linkage whose travel does not equal its joint travel (the
    finger *bodies* move opposite to the pads) -- so the joint stiffness that produces a
    given clamp force at the pad differs per arm. That is a physical difference, not a
    fudge, and it is why this is a spec field rather than one number for everybody.

    ``--gripper-ke`` on the command line overrides it."""

    collision_decompose: float | None = 0.2
    """``gs.morphs.*.decompose_robot_error_threshold`` for this arm's collision meshes.

    Genesis defaults robots to ``inf`` -- "convexify, never decompose" -- which replaces each
    ``<collision>`` mesh with its single convex hull.  For a finger that is already convex
    that is exactly right and free.  For a finger that is a concave **wedge**, the hull
    bridges the jaw slot, and the jaws bottom out on their own hulls long before the pads
    meet: measured on the reBot, the jaws stalled at a 0.60 opening fraction against the
    Newton port's 0.25 on the same pawn, i.e. **31 mm too wide**, so they pinched the piece's
    47 mm flare instead of its 11 mm neck and it slipped out on the lift. Every episode ended
    ``missed_target``; the whole arm was 0/16 on every board.

    A finite threshold means "decompose when a single hull misrepresents this mesh by more
    than this fraction", so it is self-selecting: a convex finger is left alone and a wedge is
    split. 0.2 takes the reBot's fingers to 10 hulls each and the arm from 0/8 to 8/8 on
    ``3x3``, closing to 0.233 -- Newton's number.

    Note this is the *opposite* correction from the Newton port's. There, the reBot's raw
    364k-vertex collision STLs had to be convex-hulled by hand for speed and stability;
    here Genesis has already done that, and has done too much of it."""

    clear_force_range: bool = True
    """Whether to lift the actuator force limits the asset imported with.

    Keeping them is not the conservative choice, it is an incoherent one: they size the
    real arm's motors against the asset's own kp of 10-100, while this port drives at
    600-4000, so a 10 mrad tracking error already asks for more than the limit and the
    excess is thrown away -- including the gravity compensation.  Measured here: the YAM
    imports at +-28/+-10 N m and the Piper MJCF at +-100/+-10, against the ~40 N m a 10 mrad
    error at kp=4000 asks for.  The Newton port clears the equivalent MuJoCo rows for
    exactly this reason."""

    @property
    def has_gripper(self) -> bool:
        return len(self.gripper_dofs) > 0

    @property
    def arm_dof_count(self) -> int:
        return len(self.arm_dofs)

    @property
    def gripper_dof_count(self) -> int:
        return len(self.gripper_dofs)

    @property
    def dof_count(self) -> int:
        return self.arm_dof_count + self.gripper_dof_count

    @property
    def required_links(self) -> tuple[str, ...]:
        """Every link that must survive fixed-link merging."""
        return tuple(dict.fromkeys((self.ee_link, *self.finger_links, *self.keep_links)))

    def gripper_targets(self, opened: bool) -> tuple[float, ...]:
        return self.gripper_open if opened else self.gripper_close

    def graspgen_to_ee(self, graspgen_depth: float) -> np.ndarray:
        """4x4 transform taking a GraspGen grasp pose to this arm's ``ee_link`` pose.

        GraspGen's convention is approach ``+Z``, fingers closing along ``+X``, origin at
        the gripper base with the TCP at ``+Z * depth``. Real arms disagree on both axes --
        the Franka closes along its hand's Y, the reBot reaches along its ``gripper_end``'s
        X -- so the mapping is built from the measured axes rather than assumed to be a
        rotation about one of them.

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
# Morph builders. Each returns the gs.morphs.* the arm is loaded from.
##


def _franka_morph(base_pos: Sequence[float]):
    import genesis as gs

    # The Menagerie panda.xml has a <tendon><fixed> coupling the two finger joints; the
    # newton-assets FR3 URDF has no tendon and carries a real fr3_hand_tcp body, which is
    # what makes the measured Isaac Lab TCP offset checkable against the asset. Same file
    # the Newton port loads.
    urdf = newton_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf"
    return gs.morphs.URDF(
        file=str(stage_urdf(urdf, urdf.parents[2])),
        pos=tuple(base_pos),
        fixed=True,
        # Without this, fr3_hand and fr3_hand_tcp fold into fr3_link7 -- i.e. the frame
        # the IK drives stops existing.
        links_to_keep=("fr3_hand", "fr3_hand_tcp"),
    )


def _piper_morph(base_pos: Sequence[float]):
    import genesis as gs

    # Repo-local USD rather than Menagerie agilex_piper/piper.xml: it is the asset the
    # Isaac Lab visual task spawns and the one the Newton port loads, so all three ports
    # show the same robot, and its fingers travel 50 mm each against the Menagerie model's
    # 35 mm. The Menagerie MJCF loads cleanly in Genesis too and is a drop-in alternative
    # (same link names link6/link7/link8, same joint order) if a pure-MJCF scene is wanted.
    # links_to_keep is a URDF-only field; the USD importer keeps every rigid-body prim, so
    # link6/link7/link8 arrive named by their full prim path (/piper_camera/link6). That is
    # what resolve_link()'s "unique final path component" rule is for.
    #
    # requires_jac_and_IK is NOT optional here, and it is the one morph field whose default
    # differs by importer: gs.morphs.URDF and gs.morphs.MJCF default it True, gs.morphs.USD
    # defaults it *False*. Left at the default, this arm loads, visualises and holds its home
    # pose perfectly, and then the first pick tick raises "Inverse kinematics and jacobian are
    # disabled for this entity" -- i.e. the failure lands one capability away from the change
    # that caused it. load_robot() sets it on every spec for that reason.
    return gs.morphs.USD(
        file=str(REPO_ROOT / "assets" / "piper" / "piper_camera.usd"),
        pos=tuple(base_pos),
        fixed=True,
        requires_jac_and_IK=True,
    )


def _rebot_morph(base_pos: Sequence[float]):
    import genesis as gs

    # The reBot URDF points its <collision> tags at the same full-resolution STLs as its
    # <visual> tags -- 364 392 vertices over ten links, largest 81 884. The Newton port has to
    # convex-hull them by hand (15.51 -> 3.11 ms/step); Genesis does it on import. But its
    # default goes one step too far for this arm: a single hull per wedge-shaped finger
    # bridges the jaw slot and stops the jaws 31 mm early. See
    # GenesisRobotSpec.collision_decompose, which load_robot applies to every arm.
    urdf = newton_asset("seeed_rebot_devarm") / "urdf" / "seeed_rebot_devarm.urdf"
    return gs.morphs.URDF(
        file=str(stage_urdf(urdf, urdf.parents[2])),
        pos=tuple(base_pos),
        fixed=True,
        links_to_keep=("gripper_end", "gripper_left", "gripper_right"),
    )


def _yam_morph(base_pos: Sequence[float]):
    import genesis as gs

    # No links_to_keep / merge_fixed_links: those are URDF-only morph fields, and an MJCF
    # needs neither. MuJoCo has no fixed *joints* -- a body without a joint is welded to its
    # parent and Genesis keeps it as its own link -- so link_6 and the two finger links
    # survive the import unasked. load_robot() checks that they did.
    return gs.morphs.MJCF(
        file=str(menagerie_asset("i2rt_yam") / "yam.xml"),
        pos=tuple(base_pos),
    )


def _so101_morph(base_pos: Sequence[float]):
    import genesis as gs

    return gs.morphs.MJCF(
        file=str(REPO_ROOT / "assets" / "so101" / "TheRobotStudio" / "so101_new_calib.xml"),
        pos=tuple(base_pos),
    )


def _ur10_morph(base_pos: Sequence[float]):
    import genesis as gs

    return gs.morphs.USD(
        file=str(newton_asset("universal_robots_ur10") / "usd" / "ur10_instanceable.usda"),
        pos=tuple(base_pos),
        fixed=True,
        requires_jac_and_IK=True,  # gs.morphs.USD defaults this False; see _piper_morph
    )


def _flexiv_morph(base_pos: Sequence[float]):
    import genesis as gs

    return gs.morphs.MJCF(
        file=str(menagerie_asset("flexiv_rizon4") / "flexiv_rizon4.xml"),
        pos=tuple(base_pos),
    )


##
# The table
##


ROBOTS: dict[str, GenesisRobotSpec] = {
    "franka": GenesisRobotSpec(
        key="franka",
        morph=_franka_morph,
        source="newton-assets franka_emika_panda/urdf/fr3_franka_hand.urdf",
        arm_dofs=(0, 1, 2, 3, 4, 5, 6),  # fr3_joint1..7
        gripper_dofs=(7, 8),  # fr3_finger_joint1/2
        ee_link="fr3_hand",
        finger_links=("fr3_leftfinger", "fr3_rightfinger"),
        keep_links=("fr3_hand_tcp",),
        # Isaac Lab's panda_hand -> TCP offset. Verified against this import: the URDF's own
        # fr3_hand_tcp link comes out 0.10340 m along fr3_hand's +Z, i.e. the two agree to
        # 0.06 mm.
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
    "piper": GenesisRobotSpec(
        key="piper",
        morph=_piper_morph,
        source="repo assets/piper/piper_camera.usd",
        arm_dofs=(0, 1, 2, 3, 4, 5),  # joint1..6
        gripper_dofs=(6, 7),  # joint7 [0, 0.05], joint8 [-0.05, 0]
        ee_link="link6",
        finger_links=("link7", "link8"),
        tcp_offset=(0.0, 0.0, 0.125),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(1.0, 0.0, 0.0),
        # This USD lets the fingers travel 50 mm each; the asset the Isaac Lab picking task
        # used stops at 35 mm. Command the smaller travel: the pads track the joint exactly,
        # so 0.035 reproduces the real Piper's 70 mm span.
        gripper_open=(0.035, -0.035),
        gripper_close=(0.0, 0.0),
        max_opening=0.07,
        reach=0.42,
        board_distance=0.30,
        base_pos=(-0.20, 0.0, 0.77),
        home_joint_pos=(0.0, 1.25, -1.55, 0.0, 0.95, 0.0),
    ),
    "rebot": GenesisRobotSpec(
        key="rebot",
        morph=_rebot_morph,
        source="newton-assets seeed_rebot_devarm/urdf/seeed_rebot_devarm.urdf",
        arm_dofs=(0, 1, 2, 3, 4, 5),  # joint1..6
        gripper_dofs=(6, 7),  # joint_left, joint_right, both [0, 0.05], same sign
        ee_link="gripper_end",
        finger_links=("gripper_left", "gripper_right"),
        tcp_offset=(-0.015, 0.0, 0.0),
        approach_axis=(1.0, 0.0, 0.0),
        closing_axis=(0.0, 1.0, 0.0),
        # The URDF's mimic constraint between the two fingers is not reproduced on import,
        # so both have to be commanded.
        gripper_open=(0.045, 0.045),
        gripper_close=(0.0, 0.0),
        max_opening=0.09,
        reach=0.44,
        board_distance=0.30,
        base_pos=(-0.20, 0.0, 0.77),
        # Re-solved for this asset by the Newton port: Isaac Lab's
        # (0, -1.25, -1.55, 0, -0.75, 0) was measured on the Seeed USD, whose joint frames
        # differ from this URDF's, and leaves the gripper pointing up and sideways. This
        # puts the TCP 0.24 m over the board centre, 18 degrees off straight down.
        home_joint_pos=(0.0, -1.50, -1.72, 1.48, 0.0, 0.0),
        # Measured, 8 worlds x 16 episodes at seed 0: 4x4 gives 8/16 at 300, 14/16 at 600,
        # 7/16 at 1200. The shared 300 is tuned for the Franka and is too soft for this
        # arm's wedge jaws.
        pick_gripper_kp=600.0,
    ),
    "yam": GenesisRobotSpec(
        key="yam",
        morph=_yam_morph,
        source="mujoco_menagerie i2rt_yam/yam.xml",
        arm_dofs=(0, 1, 2, 3, 4, 5),  # joint1..6
        gripper_dofs=(6, 7),  # left_finger [-0.00205, 0.037524], right_finger mirrored
        ee_link="link_6",
        # These name the finger *roots*. Genesis keeps the MJCF's linkage bodies
        # (lf_rot/lf_down, rf_rot/rf_down) as their own links where the Newton port collapses
        # them into the roots, so on this import the pads' collision geoms hang off the
        # children and these two links carry none. Nothing in the pick loop reads
        # finger_links -- it is the grasp *frame* (tcp_offset) that matters -- so this is
        # recorded rather than worked around.
        finger_links=("link_left_finger", "link_right_finger"),
        # NOT the Menagerie model's own `grasp_site` at (0, 0, 0.1347): that sits on
        # link_6's axis and the jaws do not. The slot between the jaws is centred 33 mm off
        # the axis along -Y; a grasp aimed at the axis closes on empty air. Only the lateral
        # component is corrected, the depth keeps Isaac Lab's empirical 0.13.
        tcp_offset=(0.0, -0.033, 0.13),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(1.0, 0.0, 0.0),
        # MIRRORED relative to the Isaac Lab yam.usd: here the joint extremes are open and
        # (-0.00205, +0.00205) is closed, the reverse of the lab's spec. Confirmed against
        # this import's limits: left_finger [-0.002, 0.0375], right_finger [-0.0375, 0.002].
        gripper_open=(0.037524, -0.037524),
        gripper_close=(-0.00205, 0.00205),
        # Each pad travels 39.6 mm and the pads meet at close, so the open span is the 79 mm
        # of total travel. The finger *body* origins move the opposite way to the pads on
        # this model -- measuring them gives the wrong sign.
        max_opening=0.079,
        reach=0.42,
        board_distance=0.30,
        base_pos=(-0.20, 0.0, 0.77),
        # Re-solved for this asset by the Newton port, as for the reBot.
        home_joint_pos=(0.0, 1.49, 1.67, -1.48, 0.0, 0.0),
        # Measured, 8 worlds x 16 episodes at seed 0: 4x4 gives 2/16 at 300, 7/16 at 600,
        # 10/16 at 1200; 1d gives 5, 5, 6. This is the stiffest of the four and the reason
        # pick_gripper_kp exists: the YAM's pads move through a linkage, so a given joint
        # stiffness delivers less clamp force at the pad than on a direct-drive jaw.
        pick_gripper_kp=1200.0,
    ),
    "so101": GenesisRobotSpec(
        key="so101",
        morph=_so101_morph,
        source="repo assets/so101/TheRobotStudio/so101_new_calib.xml",
        arm_dofs=(0, 1, 2, 3, 4),
        gripper_dofs=(5,),  # `gripper` is REVOLUTE on this arm
        ee_link="gripper",
        finger_links=("moving_jaw_so101_v1",),
        tcp_offset=(-0.0079, -0.0002, -0.0981),
        approach_axis=(0.0, 0.0, -1.0),
        closing_axis=(-1.0, 0.0, 0.0),
        gripper_open=(1.0,),
        gripper_close=(-0.17453,),
        max_opening=0.0978,
        reach=0.28,
        board_distance=0.19,
        base_pos=(-0.10, 0.0, 0.782),
        home_joint_pos=(0.0, -0.6, 0.9, 0.9, 0.0),
    ),
    "ur10": GenesisRobotSpec(
        key="ur10",
        morph=_ur10_morph,
        source="newton-assets universal_robots_ur10/usd/ur10_instanceable.usda",
        arm_dofs=(0, 1, 2, 3, 4, 5),
        gripper_dofs=(),  # no hand in this asset -- visualization only
        ee_link="ee_link",
        finger_links=(),
        approach_axis=(1.0, 0.0, 0.0),  # ee_link's +X points out of the tool flange
        closing_axis=(0.0, 1.0, 0.0),
        reach=0.80,
        board_distance=0.55,
        base_pos=(-0.42, 0.0, 0.77),
        home_joint_pos=(0.0, -1.2, 1.6, -1.95, -1.57, 0.0),
    ),
    "flexiv_rizon": GenesisRobotSpec(
        key="flexiv_rizon",
        morph=_flexiv_morph,
        source="mujoco_menagerie flexiv_rizon4/flexiv_rizon4.xml",
        arm_dofs=(0, 1, 2, 3, 4, 5, 6),
        gripper_dofs=(),  # no hand in this asset -- visualization only
        ee_link="link7",
        finger_links=(),
        approach_axis=(0.0, 0.0, 1.0),
        closing_axis=(1.0, 0.0, 0.0),
        reach=0.54,
        board_distance=0.38,
        base_pos=(-0.42, 0.0, 0.77),
        home_joint_pos=(0.0, -0.7, 0.0, 1.6, 0.0, 0.7, 0.0),
    ),
}

ROBOT_OPTIONS = ("franka", "piper", "rebot", "yam")
"""Arms this port supports, matching the Isaac Lab picking task's ``CHESS_ROBOTS`` and the
Newton port's list.

All four have a parallel-jaw gripper, which is what the GraspGen grasps assume. The other
three entries in :data:`ROBOTS` load and simulate but are not part of the supported set:
``ur10`` and ``flexiv_rizon`` ship no gripper at all, and the ``so101`` MJCF has a single
moving jaw rather than an opposed pair, so a GraspGen pinch does not retarget onto it. They
stay reachable through :func:`get_spec` for scene inspection; nothing here is verified
against them."""

UNSUPPORTED_ROBOTS = tuple(key for key in ROBOTS if key not in ROBOT_OPTIONS)


def get_spec(robot: str | GenesisRobotSpec) -> GenesisRobotSpec:
    if isinstance(robot, GenesisRobotSpec):
        return robot
    try:
        return ROBOTS[robot]
    except KeyError as error:
        raise ValueError(f"Unsupported robot {robot!r}. Choose one of {ROBOT_OPTIONS}.") from error


##
# Loading
##


def resolve_link(names: Sequence[str], wanted: str) -> int:
    """Index into *names* of a full link name or of a unique final path component.

    The four importers name links differently: the URDFs give ``fr3_hand``, the MJCF
    ``link_6``, and the USD importer the whole prim path ``/piper_camera/link6``. Matching
    the last ``/``-separated component keeps the specs readable without hard-coding one
    importer's naming, and it is the same rule the Newton port's ``_resolve_body`` uses.
    """
    exact = [i for i, name in enumerate(names) if name == wanted]
    if len(exact) == 1:
        return exact[0]

    tail = [i for i, name in enumerate(names) if name.rsplit("/", 1)[-1] == wanted]
    if len(tail) == 1:
        return tail[0]
    # Both counts: a name that matches two entries *exactly* leaves `tail` empty, so
    # reporting only that would say "matched 0", i.e. the opposite of what happened.
    raise KeyError(
        f"link {wanted!r} matched {len(exact)} names exactly and {len(tail)} by final "
        f"component, need exactly 1 of either, in {list(names)}"
    )


@dataclass(frozen=True)
class RobotHandle:
    """A loaded arm plus every index the scene and the controller need.

    Two index spaces are kept because Genesis uses both.  ``*_local`` indices address the
    entity (``entity.set_dofs_position(..., dofs_idx_local=...)``,
    ``entity.get_links_pos()[:, link_idx_local]``); the plain indices address the solver's
    global arrays (``scene.rigid_solver.get_links_pos(links_idx=...)``), which is how the
    pick loop reads every piece and every hand in one call instead of one call per entity.
    """

    spec: GenesisRobotSpec
    entity: Any
    """The ``gs.engine.entities.RigidEntity``."""

    link_names: tuple[str, ...]
    dofs_local: tuple[int, ...]
    """Every DOF of this entity, in local order."""

    arm_dofs_local: tuple[int, ...]
    gripper_dofs_local: tuple[int, ...]
    arm_dofs: tuple[int, ...]
    """Solver-global DOF indices of the arm, in the same order as ``arm_dofs_local``."""

    gripper_dofs: tuple[int, ...]
    ee_link_local: int
    ee_link: int
    finger_links_local: tuple[int, ...]
    finger_links: tuple[int, ...]

    @property
    def key(self) -> str:
        return self.spec.key

    def home_positions(self, opened: bool = True) -> tuple[np.ndarray, list[int]]:
        """``(values, local dof indices)`` for the home posture and gripper state."""
        values = list(self.spec.home_joint_pos)
        dofs = list(self.arm_dofs_local)
        if self.gripper_dofs_local:
            values += list(self.spec.gripper_targets(opened))
            dofs += list(self.gripper_dofs_local)
        return np.asarray(values, dtype=np.float64), dofs


def load_robot(scene: Any, robot: str | GenesisRobotSpec, base_pos: Sequence[float] | None = None) -> RobotHandle:
    """Add an arm to a Genesis scene and resolve every index the port needs.

    Must be called **before** ``scene.build()``; the gains, armature and force ranges it
    sets are applied after the build by :meth:`RobotHandle` consumers -- see
    :func:`apply_drive_settings`, which ``ChessScene.build`` calls, because Genesis has no
    pre-build hook for them.

    The DOF count and the joint limits are checked against the spec rather than trusted: a
    Genesis release that changes how fixed links merge, or an asset ref that moves, would
    otherwise shift every ``arm_dofs`` index by one and produce an arm that simply moves
    strangely.
    """
    import genesis as gs

    spec = get_spec(robot)
    position = tuple(base_pos) if base_pos is not None else spec.base_pos

    morph = spec.morph(position)
    if spec.collision_decompose is not None:
        # Applied here rather than in each morph builder so a new arm cannot forget it. Every
        # morph this port uses is a FileMorph subclass, which is where the field lives.
        morph = morph.model_copy(update={"decompose_robot_error_threshold": spec.collision_decompose})
    entity = scene.add_entity(
        morph, material=gs.materials.Rigid(gravity_compensation=spec.gravity_compensation)
    )

    if not entity.morph.requires_jac_and_IK:
        # Cheap here, expensive later: without it every inverse_kinematics call on this arm
        # raises, and the first one is several seconds into a pick episode.
        raise RuntimeError(
            f"{spec.key}: morph {type(spec.morph(position)).__name__} has requires_jac_and_IK=False, "
            "so this arm cannot be driven by IK. gs.morphs.USD defaults it False; set it in the "
            "spec's morph builder."
        )

    link_names = tuple(link.name for link in entity.links)
    dofs_local = tuple(range(entity.n_dofs))
    if entity.n_dofs != spec.dof_count:
        raise RuntimeError(
            f"{spec.key}: expected {spec.dof_count} DOFs from {spec.source}, got {entity.n_dofs}. "
            f"links: {list(link_names)}"
        )
    for name in spec.required_links:
        resolve_link(link_names, name)  # raises with a readable message if merged away

    offset = entity.dof_start
    ee_local = resolve_link(link_names, spec.ee_link)
    fingers_local = tuple(resolve_link(link_names, name) for name in spec.finger_links)
    return RobotHandle(
        spec=spec,
        entity=entity,
        link_names=link_names,
        dofs_local=dofs_local,
        arm_dofs_local=tuple(spec.arm_dofs),
        gripper_dofs_local=tuple(spec.gripper_dofs),
        arm_dofs=tuple(offset + d for d in spec.arm_dofs),
        gripper_dofs=tuple(offset + d for d in spec.gripper_dofs),
        ee_link_local=ee_local,
        ee_link=entity.link_start + ee_local,
        finger_links_local=fingers_local,
        finger_links=tuple(entity.link_start + i for i in fingers_local),
    )


def apply_drive_settings(
    handle: RobotHandle,
    arm_kp: float | None = None,
    arm_kd: float | None = None,
    gripper_kp: float | None = None,
    gripper_kd: float | None = None,
    arm_armature: float | None = None,
    gripper_armature: float | None = None,
) -> None:
    """Overwrite the imported drive gains, armature and force limits. **After** ``build()``.

    Genesis exposes these only on a built solver, so unlike the Newton port -- where they
    are builder fields -- this cannot happen at construction time.  Anything left ``None``
    falls back to the spec's single value, which is what visual inspection uses; the pick
    task passes the split arm/gripper numbers.
    """
    import numpy as np

    spec = handle.spec
    entity = handle.entity
    arm_kp = spec.target_kp if arm_kp is None else arm_kp
    arm_kd = spec.target_kd if arm_kd is None else arm_kd
    gripper_kp = arm_kp if gripper_kp is None else gripper_kp
    gripper_kd = arm_kd if gripper_kd is None else gripper_kd
    arm_armature = spec.armature if arm_armature is None else arm_armature
    gripper_armature = spec.armature if gripper_armature is None else gripper_armature

    groups = [(handle.arm_dofs_local, arm_kp, arm_kd, arm_armature)]
    if handle.gripper_dofs_local:
        groups.append((handle.gripper_dofs_local, gripper_kp, gripper_kd, gripper_armature))
    for dofs, kp, kd, armature in groups:
        count = len(dofs)
        entity.set_dofs_kp(np.full(count, kp, dtype=np.float32), dofs_idx_local=list(dofs))
        entity.set_dofs_kv(np.full(count, kd, dtype=np.float32), dofs_idx_local=list(dofs))
        entity.set_dofs_armature(np.full(count, armature, dtype=np.float32), dofs_idx_local=list(dofs))

    if spec.clear_force_range:
        count = len(handle.dofs_local)
        entity.set_dofs_force_range(
            np.full(count, -math.inf, dtype=np.float32),
            np.full(count, math.inf, dtype=np.float32),
            dofs_idx_local=list(handle.dofs_local),
        )
