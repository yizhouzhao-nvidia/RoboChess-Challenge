"""Generate Franka chess pick-and-place trajectories and record them to HDF5.

A scripted, vectorised state machine executes the move that
:class:`~robochess.tasks.manager_based.chess.mdp.commands.ChessMoveCommand` asks
for each episode: reach over the commanded piece, close on the GraspGen grasp,
lift, carry it to the target square, put it down and back off.  Isaac Lab's
:class:`~isaaclab.managers.RecorderManager` writes every episode to an HDF5
dataset, keeping only the ones the ``success`` termination confirms.

Grasps are not hand-tuned: they come from
``assets/chess/generated/<scale>/grasps/chess_grasps.json``, produced by
``lab/scripts/graspgen_chess_grasps.py``.  Each candidate is scored again at
runtime against the actual board layout, because the best grasp of a piece in
isolation may be the one that rakes the gripper through its neighbours.

.. code-block:: bash

    python lab/scripts/generate_chess_pick_demos.py --num_demos 50 --num_envs 8 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate Franka chess picking demonstrations.")
parser.add_argument("--task", type=str, default="RoboChess-Chess-Pick-IK-Abs-v0", help="Name of the task.")
parser.add_argument("--robot", type=str, default="franka", help="Arm to drive: franka, piper, rebot or yam.")
parser.add_argument(
    "--chess_scenario", choices=("pieces", "1d", "3x3", "4x4", "8x8"), default="4x4", help="Board setup to record on."
)
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments stepped in parallel.")
parser.add_argument("--num_demos", type=int, default=50, help="Number of successful demos to record.")
parser.add_argument("--max_episodes", type=int, default=0, help="Stop after this many attempts (0 = unlimited).")
parser.add_argument(
    "--dataset_file", type=str, default="./lab/datasets/franka_chess_pick.hdf5", help="Where to write the dataset."
)
parser.add_argument(
    "--export_mode",
    choices=("succeeded_only", "all", "separate"),
    default="succeeded_only",
    help="Which episodes the recorder writes out.",
)
parser.add_argument("--grasp_file", type=str, default=None, help="Override the GraspGen grasp JSON.")
parser.add_argument("--num_grasp_candidates", type=int, default=12, help="GraspGen candidates considered per piece.")
parser.add_argument("--num_yaw_candidates", type=int, default=16, help="Yaw samples for pieces of revolution.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--balance_kinds",
    action="store_true",
    help="Steer each episode toward the piece kind with the fewest recorded demos, so a short run still covers pawn, rook, knight, bishop, queen and king.",
)
parser.add_argument("--debug", action="store_true", help="Trace env 0 through every phase of the state machine.")
parser.add_argument("--video", action="store_true", help="Record a video of the first episodes.")
parser.add_argument("--video_length", type=int, default=900, help="Video length in environment steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import json
import math
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
import robochess.tasks  # noqa: F401
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg

from robochess.tasks.manager_based.chess.franka_chess_env_cfg import GENERATED_ASSET_DIR, TABLE_TOP_Z
from robochess.tasks.manager_based.chess.mdp import piece_grasped
from robochess.tasks.manager_based.chess.robot_configs import ChessRobotSpec

##
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


class ChessPickPolicy:
    """Vectorised scripted controller producing absolute panda_hand pose actions."""

    def __init__(self, env, grasps: GraspLibrary, piece_geometry: dict[str, dict], spec: ChessRobotSpec, args):
        self.env = env
        self.spec = spec
        self.grasps = grasps
        self.args = args
        self.device = env.device
        self.num_envs = env.num_envs

        self.command = env.command_manager.get_term("chess_move")
        self.piece_names = list(self.command.cfg.piece_names)
        self.piece_kinds = list(self.command.cfg.piece_kinds)
        self.robot = env.scene["robot"]

        self.piece_radius = torch.tensor(
            [piece_geometry[kind]["base_diameter"] / 2.0 for kind in self.piece_kinds], device=self.device
        )
        self.piece_height = torch.tensor(
            [piece_geometry[kind]["height"] for kind in self.piece_kinds], device=self.device
        )
        self.probe_points = gripper_probe_points(spec, self.device)

        # The carried piece hangs below the hand with its base at the lift height, so
        # the carry has to clear the tallest piece still standing on the board.
        # Clear the tallest piece on the board -- but not beyond what this arm can
        # hold. A carry sized for a Panda puts a reBot's wrist outside its workspace,
        # where the IK gives up and the piece is dropped mid-transfer.
        wanted = float(self.piece_height.max()) + CARRY_CLEARANCE
        self.lift_height = max(MIN_LIFT_HEIGHT, min(wanted, CARRY_REACH_FRACTION * spec.reach))
        print(f"[INFO] Carrying pieces {self.lift_height * 1000:.0f} mm above the board")

        self.phase_steps = torch.tensor(
            [max(1, round(phase.duration / env.step_dt)) for phase in PHASES], device=self.device
        )
        self.phase_deadline = self.phase_steps + torch.tensor(
            [max(0, round(phase.settle_timeout / env.step_dt)) for phase in PHASES], device=self.device
        )
        self.pos_tolerance = torch.tensor([phase.pos_tolerance for phase in PHASES], device=self.device)
        self.rot_tolerance = torch.tensor([phase.rot_tolerance for phase in PHASES], device=self.device)
        self.phase_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        pose_shape = (self.num_envs, 3)
        self.goal_pos = {name: torch.zeros(pose_shape, device=self.device) for name in _GOAL_NAMES}
        self.goal_quat = {name: torch.zeros(self.num_envs, 4, device=self.device) for name in _GOAL_NAMES}
        for quat in self.goal_quat.values():
            quat[:, 3] = 1.0
        self.start_pos = torch.zeros(pose_shape, device=self.device)
        self.start_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.start_quat[:, 3] = 1.0
        self.command_pos = self.start_pos.clone()
        self.command_quat = self.start_quat.clone()
        self.hand_in_piece = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        # The environment auto-resets inside env.step(), which resamples the command
        # before the caller sees the `done` flags. Anything read from the command
        # afterwards describes the *next* episode, so the kind under test has to be
        # snapshotted when the episode is planned.
        self.planned_kind: list[str] = ["" for _ in range(self.num_envs)]

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Re-plan the grasp and phase schedule for the given environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        self.phase_index[env_ids] = 0
        self.phase_step[env_ids] = 0

        hand_pos_b, hand_quat_b = self._hand_pose_b()
        self.command_pos[env_ids] = hand_pos_b[env_ids]
        self.command_quat[env_ids] = hand_quat_b[env_ids]
        self.start_pos[env_ids] = hand_pos_b[env_ids]
        self.start_quat[env_ids] = hand_quat_b[env_ids]

        for env_id in env_ids.tolist():
            self._plan_grasp(env_id)
            self.planned_kind[env_id] = self.piece_kinds[int(self.command.piece_index[env_id])]

    def compute(self) -> torch.Tensor:
        """Advance the state machine one control step and return the action tensor."""
        interp_steps = self.phase_steps[self.phase_index.clamp(max=len(PHASES) - 1)]
        tau = (self.phase_step.float() / interp_steps.float()).clamp(max=1.0)

        goal_pos = torch.zeros_like(self.start_pos)
        goal_quat = torch.zeros_like(self.start_quat)
        gripper = torch.zeros(self.num_envs, 1, device=self.device)
        for index, phase in enumerate(PHASES):
            mask = self.phase_index == index
            if not mask.any():
                continue
            goal_pos[mask] = self.goal_pos[phase.goal][mask]
            goal_quat[mask] = self.goal_quat[phase.goal][mask]
            gripper[mask, 0] = phase.gripper
        finished = self.phase_index >= len(PHASES)
        if finished.any():
            last = PHASES[-1]
            goal_pos[finished] = self.goal_pos[last.goal][finished]
            goal_quat[finished] = self.goal_quat[last.goal][finished]
            gripper[finished, 0] = last.gripper

        self.command_pos = torch.lerp(self.start_pos, goal_pos, tau.unsqueeze(-1))
        self.command_quat = slerp(self.start_quat, goal_quat, tau)


        self._advance(interp_steps)
        return torch.cat([self.command_pos, self.command_quat, gripper], dim=-1)

    """
    Internals.
    """

    def report(self, env_id: int = 0) -> str:
        """One-line snapshot of env ``env_id`` for --debug."""
        index = int(self.phase_index[env_id])
        phase = PHASES[index].name if index < len(PHASES) else "done"
        hand_pos_b, hand_quat_b = self._hand_pose_b()
        tracking = float(torch.norm(hand_pos_b[env_id] - self.command_pos[env_id]))
        rotation = float(math_utils.quat_error_magnitude(hand_quat_b[env_id : env_id + 1], self.command_quat[env_id : env_id + 1]))

        piece_index = int(self.command.piece_index[env_id])
        piece_pos = self._piece_poses_w()[0][env_id, piece_index]
        target = self.command.target_pos_w[env_id]
        finger_ids = self.robot.find_joints(list(self.spec.gripper_joints))[0]
        gap = float(self.robot.data.joint_pos.torch[env_id, finger_ids].sum())
        return (
            f"[env{env_id}] {phase:<10s} {self.piece_kinds[piece_index]:<6s}"
            f" track={tracking * 1000:6.1f}mm/{math.degrees(rotation):5.1f}deg"
            f" gap={gap * 1000:5.1f}mm hold={int(self._holding()[env_id])}"
            f" piece_z={float(piece_pos[2] - TABLE_TOP_Z) * 1000:6.1f}mm"
            f" to_target={float(torch.norm(piece_pos[:2] - target[:2])) * 1000:6.1f}mm"
            f" place_goal={self.goal_pos['place'][env_id].cpu().numpy().round(3)}"
        )

    def _advance(self, interp_steps: torch.Tensor) -> None:
        self.phase_step += 1
        active = self.phase_index < len(PHASES)
        index = self.phase_index.clamp(max=len(PHASES) - 1)

        hand_pos_b, hand_quat_b = self._hand_pose_b()
        pos_error = torch.norm(hand_pos_b - self.command_pos, dim=-1)
        rot_error = math_utils.quat_error_magnitude(hand_quat_b, self.command_quat)
        arrived = (pos_error < self.pos_tolerance[index]) & (rot_error < self.rot_tolerance[index])

        rollover = active & (self.phase_step >= interp_steps) & (arrived | (self.phase_step >= self.phase_deadline[index]))
        if self.args.debug and bool(rollover[0]):
            print(self.report(0))
        if not rollover.any():
            return
        # The place pose depends on how the piece actually sits in the hand, so it is
        # derived from the measured hand-to-piece transform -- but only for envs that
        # really are holding the piece, since deriving it from a failed grasp aims the
        # arm at a pose metres away and turns a quiet failure into a thrashing one.
        # It is refreshed at three points rather than one: the fingers may still be
        # travelling when `close` ends, and the piece can settle or slip during the
        # lift and the carry.
        holding = self._holding()
        for phase_name in _CAPTURE_AFTER:
            recompute = rollover & (self.phase_index == _PHASE_BY_NAME[phase_name]) & holding
            if recompute.any():
                self._capture_grasp(recompute.nonzero().flatten())

        self.start_pos[rollover] = self.command_pos[rollover]
        self.start_quat[rollover] = self.command_quat[rollover]
        self.phase_index[rollover] += 1
        self.phase_step[rollover] = 0

    def _holding(self) -> torch.Tensor:
        """Whether each env currently has the commanded piece between its fingers.

        Delegates to the environment's own observation term so the state machine and
        the ``success`` termination can never disagree about what "holding" means.
        """
        return piece_grasped(self.env, dict(self.spec.open_command), dict(self.spec.close_command)).squeeze(-1)

    def _hand_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current end-effector pose in the robot base frame -- what the IK action commands."""
        body_index = self.robot.find_bodies(self.spec.ee_body)[0][0]
        return math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.robot.data.body_pos_w.torch[:, body_index],
            self.robot.data.body_quat_w.torch[:, body_index],
        )

    def _piece_poses_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.stack([self.env.scene[name].data.root_pos_w.torch for name in self.piece_names], dim=1)
        quaternions = torch.stack([self.env.scene[name].data.root_quat_w.torch for name in self.piece_names], dim=1)
        return positions, quaternions

    def _plan_grasp(self, env_id: int) -> None:
        """Pick the GraspGen candidate that the current board leaves room for."""
        piece_index = int(self.command.piece_index[env_id])
        kind = self.piece_kinds[piece_index]
        candidates, scores = self.grasps.candidates(kind, self.args.num_yaw_candidates)

        positions, quaternions = self._piece_poses_w()
        piece_pose_w = pose_to_matrix(positions[env_id, piece_index : piece_index + 1], quaternions[env_id, piece_index : piece_index + 1])[0]
        hand_w = piece_pose_w.unsqueeze(0) @ candidates  # (C, 4, 4)

        # Score every candidate by how deep the open fingers dip into a neighbour,
        # modelling each other piece as an upright cylinder.
        probes = self.probe_points.unsqueeze(0)  # (1, P, 3)
        grasp_points = probes @ hand_w[:, :3, :3].transpose(1, 2) + hand_w[:, None, :3, 3]
        standoff = hand_w[:, :3, 2] * APPROACH_STANDOFF
        approach_points = grasp_points - standoff[:, None, :]
        probe_world = torch.cat([grasp_points, approach_points], dim=1)  # (C, 2P, 3)

        others = [index for index in range(len(self.piece_names)) if index != piece_index]
        penetration = torch.zeros(len(candidates), device=self.device)
        if others:
            centers = positions[env_id, others]  # (O, 3)
            radii = self.piece_radius[others]
            tops = centers[:, 2] + self.piece_height[others]
            radial = torch.norm(probe_world[:, :, None, :2] - centers[None, None, :, :2], dim=-1)
            inside_column = (radii[None, None, :] + _CLEARANCE_MARGIN) - radial
            below_top = tops[None, None, :] + _CLEARANCE_MARGIN - probe_world[:, :, None, 2]
            overlap = torch.minimum(inside_column, below_top).clamp(min=0.0)
            penetration = overlap.amax(dim=(1, 2))

        best = torch.argmax(scores - _COLLISION_WEIGHT * penetration)
        grasp_w = hand_w[best]

        root_pos = self.robot.data.root_pos_w.torch[env_id : env_id + 1]
        root_quat = self.robot.data.root_quat_w.torch[env_id : env_id + 1]
        grasp_pos_b, grasp_quat_b = math_utils.subtract_frame_transforms(
            root_pos, root_quat, grasp_w[None, :3, 3], math_utils.quat_from_matrix(grasp_w[None, :3, :3])
        )
        approach_b = math_utils.matrix_from_quat(grasp_quat_b)[:, :3, 2]

        self.goal_pos["grasp"][env_id] = grasp_pos_b[0]
        self.goal_quat["grasp"][env_id] = grasp_quat_b[0]
        self.goal_pos["pre_grasp"][env_id] = (grasp_pos_b - approach_b * APPROACH_STANDOFF)[0]
        self.goal_quat["pre_grasp"][env_id] = grasp_quat_b[0]
        self.goal_pos["lift"][env_id] = grasp_pos_b[0] + torch.tensor([0.0, 0.0, self.lift_height], device=self.device)
        self.goal_quat["lift"][env_id] = grasp_quat_b[0]
        # Provisional place targets; refined in _capture_grasp once the piece is held.
        for name in ("place", "pre_place"):
            self.goal_pos[name][env_id] = self.goal_pos["lift"][env_id]
            self.goal_quat[name][env_id] = grasp_quat_b[0]

    def _capture_grasp(self, env_ids: torch.Tensor) -> None:
        """Freeze the hand-to-piece transform and turn the target square into hand poses."""
        positions, quaternions = self._piece_poses_w()
        index = self.command.piece_index[env_ids]
        piece_pos = positions[env_ids, index]
        piece_quat = quaternions[env_ids, index]
        piece_w = pose_to_matrix(piece_pos, piece_quat)

        hand_pos_b, hand_quat_b = self._hand_pose_b()
        root_pos = self.robot.data.root_pos_w.torch[env_ids]
        root_quat = self.robot.data.root_quat_w.torch[env_ids]
        hand_pos_w, hand_quat_w = math_utils.combine_frame_transforms(
            root_pos, root_quat, hand_pos_b[env_ids], hand_quat_b[env_ids]
        )
        hand_w = pose_to_matrix(hand_pos_w, hand_quat_w)
        self.hand_in_piece[env_ids] = torch.linalg.inv(piece_w) @ hand_w

        # Put the piece down upright, keeping the yaw it had when it was picked up.
        target_w = self.command.target_pos_w[env_ids].clone()
        target_w[:, 2] += PLACE_CLEARANCE
        upright_quat = math_utils.yaw_quat(piece_quat)
        place_piece_w = pose_to_matrix(target_w, upright_quat)
        place_hand_w = place_piece_w @ self.hand_in_piece[env_ids]

        place_pos_b, place_quat_b = math_utils.subtract_frame_transforms(
            root_pos, root_quat, place_hand_w[:, :3, 3], math_utils.quat_from_matrix(place_hand_w[:, :3, :3])
        )
        self.goal_pos["place"][env_ids] = place_pos_b
        self.goal_quat["place"][env_ids] = place_quat_b
        self.goal_pos["pre_place"][env_ids] = place_pos_b + torch.tensor(
            [0.0, 0.0, PLACE_APPROACH_HEIGHT], device=self.device
        )
        self.goal_quat["pre_place"][env_ids] = place_quat_b


_GOAL_NAMES = ("pre_grasp", "grasp", "lift", "pre_place", "place")
_PHASE_BY_NAME = {phase.name: index for index, phase in enumerate(PHASES)}
_CAPTURE_AFTER = ("close", "lift", "transfer")
_CLEARANCE_MARGIN = 0.004
_COLLISION_WEIGHT = 40.0

_EXPORT_MODES = {
    "succeeded_only": DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
    "all": DatasetExportMode.EXPORT_ALL,
    "separate": DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES,
}


def main():
    dataset_path = Path(args_cli.dataset_file).resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.set_robot(args_cli.robot)
    env_cfg.set_chess_scenario(args_cli.chess_scenario)
    env_cfg.seed = args_cli.seed

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
    env_cfg.recorders.dataset_filename = dataset_path.stem
    env_cfg.recorders.dataset_export_mode = _EXPORT_MODES[args_cli.export_mode]

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(dataset_path.parent / "videos"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    inner = env.unwrapped

    asset_dir = GENERATED_ASSET_DIR / env_cfg.piece_scale_tag
    grasp_file = Path(args_cli.grasp_file) if args_cli.grasp_file else asset_dir / "grasps" / "chess_grasps.json"
    if not grasp_file.exists():
        raise FileNotFoundError(
            f"Missing GraspGen grasps at {grasp_file}. Run:\n"
            f"    /home/yizhou/Projects/GraspGen/.venv/bin/python lab/scripts/graspgen_chess_grasps.py"
        )
    piece_geometry = json.loads((asset_dir / "pieces.json").read_text())["pieces"]
    spec = env_cfg.chess_robot()
    grasps = GraspLibrary(grasp_file, spec, inner.device, args_cli.num_grasp_candidates)

    env.reset()
    policy = ChessPickPolicy(inner, grasps, piece_geometry, spec, args_cli)
    policy.reset()

    attempts = 0
    recorded = 0
    failures: dict[str, int] = {}
    attempted_kinds: dict[str, int] = {kind: 0 for kind in set(policy.piece_kinds)}
    print(f"[INFO] Robot: {args_cli.robot}  ({spec.ee_body}, stroke {spec.max_opening * 1000:.0f} mm)")
    print(f"[INFO] Recording to {dataset_path} (mode={args_cli.export_mode}); target {args_cli.num_demos} demos.")
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            actions = policy.compute()
            _, _, terminated, truncated, _ = env.step(actions)

            done = (terminated | truncated).nonzero().flatten()
            if len(done) > 0:
                if args_cli.debug and bool(done.eq(0).any()):
                    reasons = [
                        name
                        for name in inner.termination_manager.active_terms
                        if bool(inner.termination_manager.get_term(name)[0])
                    ]
                    print(f"[env0] episode end after {int(policy.phase_index[0])} phases: {reasons or ['unknown']}\n")
                for env_id in done.tolist():
                    if bool(truncated[env_id]):
                        failures["timed_out"] = failures.get("timed_out", 0) + 1
                    # Count against the kind planned for this episode, not whatever the
                    # command holds now -- the env has already reset and resampled.
                    attempted_kinds[policy.planned_kind[env_id]] += 1
                attempts += len(done)
                if args_cli.balance_kinds:
                    policy.command.prefer_kinds(
                        done, least_attempted_kinds(attempted_kinds, policy.piece_kinds)
                    )
                policy.reset(done)
                if inner.recorder_manager.exported_successful_episode_count > recorded:
                    recorded = inner.recorder_manager.exported_successful_episode_count
                    print(f"[INFO] {recorded}/{args_cli.num_demos} demos recorded ({attempts} attempts)")

            if recorded >= args_cli.num_demos:
                print(f"[INFO] Reached {recorded} successful demos.")
                break
            if args_cli.max_episodes and attempts >= args_cli.max_episodes:
                print(f"[WARN] Stopping after {attempts} attempts with {recorded} successes.")
                break

    if args_cli.balance_kinds:
        print("[INFO] attempts by piece kind: " + ", ".join(f"{k}={v}" for k, v in sorted(attempted_kinds.items())))
        print("[INFO] for per-kind coverage read the dataset: lab/scripts/dataset_summary.py")
    success_rate = recorded / attempts if attempts else 0.0
    print(f"\n[INFO] {recorded} successes / {attempts} attempts ({success_rate:.0%})")
    if failures:
        print("[INFO] failures by cause: " + ", ".join(f"{k}={v}" for k, v in sorted(failures.items())))
    print(f"[INFO] Dataset: {dataset_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
