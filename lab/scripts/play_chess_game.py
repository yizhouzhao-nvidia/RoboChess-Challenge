"""Two robot arms playing a chess variant against each other, and recording it.

Each arm owns one colour. The rules engine
(:mod:`robochess.tasks.manager_based.chess.chess_rules`) chooses a legal move, and the
arm belonging to the side to move executes it as a pick-and-place; the other arm holds
station. A capture is two pick-and-places -- the captured piece goes to the off-board
tray first, because dropping a piece onto an occupied square just knocks both over.

.. code-block:: bash

    python lab/scripts/play_chess_game.py --headless --chess_scenario 3x3 --num_games 3
    python lab/scripts/play_chess_game.py --headless --chess_scenario 1d --white franka --black franka
    python lab/scripts/play_chess_game.py --headless --chess_scenario minichess --max_plies 16

One game is one recorded episode. Unlike the single-arm task, success is not a scene
predicate -- "checkmate" is a fact about the rules, not about where the pieces are --
so the driver flags it through the recorder API directly.

Single environment by design: a game is a long sequential dependency (ply *n* depends
on where ply *n-1* actually left the piece), so batching buys little and makes the
bookkeeping considerably easier to get wrong.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play chess variants with two robot arms.")
parser.add_argument("--task", type=str, default="RoboChess-Chess-Game-IK-Abs-v0")
parser.add_argument("--white", type=str, default="franka", help="Arm playing white.")
parser.add_argument("--black", type=str, default="franka", help="Arm playing black.")
parser.add_argument(
    "--chess_scenario", choices=("1d", "3x3", "minichess"), default="3x3", help="Which variant to play."
)
parser.add_argument("--num_games", type=int, default=3, help="Games to record.")
parser.add_argument("--max_plies", type=int, default=12, help="Ply cap before a game is abandoned.")
parser.add_argument(
    "--policy",
    choices=("greedy", "random"),
    default="greedy",
    help="Move choice: 'greedy' prefers mate then captures, 'random' picks uniformly.",
)
parser.add_argument(
    "--dataset_file", type=str, default="./lab/datasets/chess_game.hdf5", help="Where to write the dataset."
)
parser.add_argument("--grasp_file", type=str, default=None, help="Override the GraspGen grasp JSON.")
parser.add_argument("--num_grasp_candidates", type=int, default=12)
parser.add_argument("--num_yaw_candidates", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug", action="store_true", help="Print every phase transition.")
parser.add_argument("--video", action="store_true", help="Record a video of the games.")
parser.add_argument("--video_length", type=int, default=4000)
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
import random
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
import robochess.tasks  # noqa: F401
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg

from robochess.tasks.manager_based.chess.chess_rules import Move, Position
from robochess.tasks.manager_based.chess.franka_chess_env_cfg import GENERATED_ASSET_DIR, TABLE_TOP_Z
from robochess.tasks.manager_based.chess.motion import (
    APPROACH_STANDOFF,
    CARRY_CLEARANCE,
    CARRY_REACH_FRACTION,
    MIN_LIFT_HEIGHT,
    PLACE_APPROACH_HEIGHT,
    PLACE_CLEARANCE,
    GraspLibrary,
    gripper_probe_points,
    matrix_to_pose,
    pose_to_matrix,
    slerp,
)
from robochess.tasks.manager_based.chess.robot_configs import ChessRobotSpec

GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0

PHASES = (
    # name, seconds of interpolation, gripper, waypoint, position tolerance [m]
    ("pre_grasp", 1.8, GRIPPER_OPEN, "pre_grasp", 0.006),
    ("descend", 1.2, GRIPPER_OPEN, "grasp", 0.003),
    ("close", 0.9, GRIPPER_CLOSE, "grasp", 0.003),
    ("lift", 1.2, GRIPPER_CLOSE, "lift", 0.008),
    ("transfer", 2.2, GRIPPER_CLOSE, "pre_place", 0.008),
    ("place", 1.4, GRIPPER_CLOSE, "place", 0.004),
    ("release", 0.6, GRIPPER_OPEN, "place", 0.004),
    ("retreat", 1.0, GRIPPER_OPEN, "pre_place", 0.02),
)
SETTLE_SECONDS = 2.0
"""Extra time a phase may spend waiting to arrive before its deadline fires."""


class ArmController:
    """Drives one arm through a pick-and-place, one waypoint at a time."""

    def __init__(self, env, player: str, spec: ChessRobotSpec, grasps: GraspLibrary, tallest_piece: float):
        self.env = env
        self.player = player
        self.spec = spec
        self.grasps = grasps
        self.tallest_piece = tallest_piece
        self.max_carry = CARRY_REACH_FRACTION * spec.reach
        self.probes = gripper_probe_points(spec, env.device)
        self.fail_reason: str | None = None
        self.device = env.device
        self.robot = env.scene[f"robot_{player}"]
        self.settle_scale = spec.settle_scale

        self.waypoints: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.phase = len(PHASES)
        self.phase_step = 0
        self.start_pose: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------ geometry
    def hold_command(self) -> torch.Tensor:
        """Command that keeps the arm where it is: its own body pose, in its own frame."""
        pos, quat = self.body_pose_w()
        return self._to_base_frame(pos, quat)

    @property
    def body_index(self) -> int:
        return self.robot.find_bodies(self.spec.ee_body)[0][0]

    def body_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose of the body the IK drives -- read from physics, never from USD."""
        index = self.body_index
        return self.robot.data.body_pos_w.torch[:, index], self.robot.data.body_quat_w.torch[:, index]

    def _to_base_frame(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, pos_w, quat_w
        )
        return torch.cat([pos_b, quat_b], dim=-1)

    def _reach_of(self, pos_w: torch.Tensor) -> torch.Tensor:
        base = self.robot.data.root_pos_w.torch
        return torch.norm(pos_w[:, :2] - base[:, :2], dim=-1)

    def clear_of_neighbours(self, pos: torch.Tensor, quat: torch.Tensor, neighbours) -> torch.Tensor:
        """Which candidate grasps keep the open fingers out of every other piece.

        Without this the highest-scoring grasp wins outright, and on a packed board that
        routinely means descending with a finger through the piece next door: at a 84 mm
        pitch a king's 59 mm base leaves 25 mm for two ~13 mm fingers. It is why 1D chess
        and minichess knocked something over in every single game while Hexapawn -- short
        pieces, well spread -- mostly survived.
        """
        if neighbours is None:
            return torch.ones(pos.shape[0], dtype=torch.bool, device=self.device)
        centres, radii, tops = neighbours
        rotation = math_utils.matrix_from_quat(quat)                          # (C, 3, 3)
        probes = (rotation @ self.probes.T).transpose(1, 2) + pos.unsqueeze(1)  # (C, P, 3)
        planar = torch.norm(probes[:, :, None, :2] - centres[None, None, :, :2], dim=-1)
        hits = (planar < radii[None, None, :]) & (probes[:, :, None, 2] < tops[None, None, :])
        return ~hits.any(dim=2).any(dim=1)

    def choose_grasp(self, piece_pos_w: torch.Tensor, piece_quat_w: torch.Tensor, kind: str, neighbours=None):
        """Best-scoring reachable grasp for the piece as it currently stands.

        Returns the ee-body pose of the grasp, or ``None`` if no candidate is within
        reach -- which is a real outcome on the wider boards and has to be reported
        rather than silently clamped.
        """
        hand_in_piece, scores = self.grasps.candidates(kind, args_cli.num_yaw_candidates)
        piece = pose_to_matrix(piece_pos_w, piece_quat_w)                     # (1, 4, 4)
        world = piece @ hand_in_piece                                        # (C, 4, 4)
        pos, quat = matrix_to_pose(world.reshape(-1, 4, 4))

        reach = self._reach_of(pos)
        approach_w = math_utils.matrix_from_quat(quat) @ torch.tensor(
            self.spec.approach_axis, device=self.device, dtype=torch.float32
        )
        # Keep grasps that point downwards: an upward-pointing hand means the arm has
        # to come from under the table.
        downward = approach_w[:, 2] < -0.2
        feasible = (reach <= self.spec.reach * 0.95) & downward & self.clear_of_neighbours(pos, quat, neighbours)
        if not bool(feasible.any()):
            return None
        ranked = torch.where(feasible, scores, torch.full_like(scores, -1e6))
        best = int(torch.argmax(ranked))
        return pos[best: best + 1], quat[best: best + 1]

    def plan(self, piece_pos_w, piece_quat_w, kind: str, target_xyz, neighbours=None) -> bool:
        """Build the waypoints for moving one piece to one destination."""
        self.fail_reason = None
        # Toppled has to be judged from orientation, not height. The asset bake puts each
        # piece's origin at its *base*, so a fallen piece sits at the same height as a
        # standing one and a height test calls every piece flat -- including on ply 1,
        # before an arm has moved. Same 30 deg test the single-arm task terminates on.
        up = math_utils.quat_apply(piece_quat_w, torch.tensor([[0.0, 0.0, 1.0]], device=self.device))
        if float(up[0, 2]) < math.cos(math.radians(30.0)):
            self.fail_reason = f"piece is lying down (up-axis z={float(up[0, 2]):.2f})"
            return False
        grasp = self.choose_grasp(piece_pos_w, piece_quat_w, kind, neighbours)
        if grasp is None:
            self.fail_reason = "no grasp that is reachable, downward and clear of neighbours"
            return False
        grasp_pos, grasp_quat = grasp

        approach = math_utils.matrix_from_quat(grasp_quat) @ torch.tensor(
            self.spec.approach_axis, device=self.device, dtype=torch.float32
        )
        target = torch.tensor([target_xyz], device=self.device, dtype=torch.float32)
        target = target + self.env.scene.env_origins

        # The piece hangs below the hand by however far up its shaft it was gripped, so
        # clearing the tallest piece on the board needs that height *added* to it. Using
        # the board's tallest piece alone leaves a carried pawn's base level with the
        # tops of the pawns it flies over, and it drags them off their squares -- which
        # is what made games abort a few plies in, on pieces that had been knocked flat.
        grip_height = float(grasp_pos[0, 2]) - float(piece_pos_w[0, 2])
        carry_z = TABLE_TOP_Z + min(
            self.tallest_piece + grip_height + CARRY_CLEARANCE, self.max_carry
        )

        self.waypoints = {
            "grasp": (grasp_pos, grasp_quat),
            "pre_grasp": (grasp_pos - approach * APPROACH_STANDOFF, grasp_quat),
            "lift": (torch.cat([grasp_pos[:, :2], grasp_pos[:, 2:] * 0 + carry_z], -1), grasp_quat),
            # Carried at the same height all the way across: the destination is empty,
            # but everything between here and there is not.
            "pre_place": (torch.cat([target[:, :2], target[:, 2:] * 0 + carry_z], -1), grasp_quat),
            "place": (
                torch.cat(
                    [target[:, :2], target[:, 2:] * 0 + TABLE_TOP_Z + grip_height + PLACE_CLEARANCE], -1
                ),
                grasp_quat,
            ),
        }
        if float(self._reach_of(self.waypoints["pre_place"][0])) > self.spec.reach * 0.98:
            self.fail_reason = "destination out of reach"
            return False

        self.phase = 0
        self.phase_step = 0
        self.start_pose = None
        return True

    # ------------------------------------------------------------------ execution
    @property
    def busy(self) -> bool:
        return self.phase < len(PHASES)

    def step(self) -> tuple[torch.Tensor, float, str | None]:
        """Advance one control step. Returns (arm command, gripper, phase just finished)."""
        if not self.busy:
            return self.hold_command(), GRIPPER_OPEN, None

        name, seconds, gripper, waypoint, tolerance = PHASES[self.phase]
        steps = max(1, round(seconds / self.env.step_dt))
        deadline = steps + max(0, round(SETTLE_SECONDS * self.settle_scale / self.env.step_dt))

        goal_pos, goal_quat = self.waypoints[waypoint]
        if self.start_pose is None:
            pos_w, quat_w = self.body_pose_w()
            self.start_pose = (pos_w.clone(), quat_w.clone())

        tau = torch.tensor([min(1.0, (self.phase_step + 1) / steps)], device=self.device)
        pos = self.start_pose[0] + (goal_pos - self.start_pose[0]) * tau.unsqueeze(-1)
        quat = slerp(self.start_pose[1], goal_quat, tau)

        body_pos, _ = self.body_pose_w()
        error = float(torch.norm(body_pos - goal_pos, dim=-1))
        arrived = error < tolerance

        self.phase_step += 1
        finished = None
        if self.phase_step >= steps and (arrived or self.phase_step >= deadline):
            finished = name
            self.phase += 1
            self.phase_step = 0
            self.start_pose = None
            if args_cli.debug:
                print(f"      [{self.player}] {name:10s} error={error * 1000:6.1f}mm")

        return self._to_base_frame(pos, quat), gripper, finished


def toppled_pieces(inner, piece_names: list[str], position: Position) -> list[str]:
    """Pieces still in play that are no longer standing up.

    A game whose moves are all legal can still be worthless: if the arm barges a
    neighbour over on its way past, the board no longer matches the position the rules
    engine believes in, and the demonstration teaches a policy to do the same. The
    single-arm task rejects these through its ``board_disturbed`` termination; the game
    env has no terminations by design, so the check lives here instead.

    Captured pieces are skipped -- they are lying in the tray, which is where they belong.
    """
    knocked = []
    up_axis = torch.tensor([[0.0, 0.0, 1.0]], device=inner.device)
    limit = math.cos(math.radians(30.0))
    for index, piece in enumerate(position.pieces):
        if not piece.alive:
            continue
        quat = inner.scene[piece_names[index]].data.root_quat_w.torch
        if float(math_utils.quat_apply(quat, up_axis)[0, 2]) < limit:
            knocked.append(piece_names[index])
    return knocked


def choose_move(position: Position, rng: random.Random) -> Move:
    """Pick a legal move. Greedy prefers a mate, then a capture, then anything."""
    moves = position.legal_moves()
    if args_cli.policy == "random":
        return rng.choice(moves)
    mating = [m for m in moves if position.apply(m).result() == position.side_to_move]
    if mating:
        return rng.choice(mating)
    captures = [m for m in moves if m.captured is not None]
    return rng.choice(captures or moves)


def main():
    dataset_path = Path(args_cli.dataset_file).resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.set_players(args_cli.white, args_cli.black)
    env_cfg.set_chess_scenario(args_cli.chess_scenario)
    env_cfg.max_plies = args_cli.max_plies
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = env_cfg.episode_budget()

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
    env_cfg.recorders.dataset_filename = dataset_path.stem
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

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

    layout = env_cfg.chess_layout()
    asset_dir = GENERATED_ASSET_DIR / env_cfg.piece_scale_tag
    grasp_file = Path(args_cli.grasp_file) if args_cli.grasp_file else asset_dir / "grasps" / "chess_grasps.json"
    geometry = json.loads((asset_dir / "pieces.json").read_text())["pieces"]

    tallest = max(geometry[spec.kind]["height"] for spec in layout.pieces)
    piece_names = [spec.name for spec in layout.pieces]
    trays = {player: env_cfg.capture_tray_positions(layout, player) for player in ('white', 'black')}

    unreachable = env_cfg.unreachable_squares(layout)
    for player, squares in unreachable.items():
        if squares:
            print(f"[WARN] {player} ({env_cfg.player_spec(player).key}) cannot reach {len(squares)} square(s): {squares}")
    for player, count in env_cfg.unreachable_tray_slots(layout).items():
        if count:
            print(f"[WARN] {player} cannot reach {count} of its own capture-tray slots")

    arms = {}
    for player in ("white", "black"):
        spec = env_cfg.player_spec(player)
        arms[player] = ArmController(
            inner, player, spec,
            GraspLibrary(grasp_file, spec, inner.device, args_cli.num_grasp_candidates), tallest,
        )
        print(f"[INFO] {player}: {spec.key} (carry ceiling {arms[player].max_carry * 1000:.0f} mm)")

    env.reset()
    rng = random.Random(args_cli.seed)
    games_recorded, attempts, results = 0, 0, {}
    # A game that aborts is retried, but not forever: if the pair simply cannot play
    # this board, say so rather than spinning.
    max_attempts = max(4, args_cli.num_games * 4)

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while games_recorded < args_cli.num_games and attempts < max_attempts and simulation_app.is_running():
            attempts += 1
            position = Position.from_layout(layout)
            plies, aborted = 0, None
            pending: list[tuple[str, int, tuple]] = []   # (player, piece index, destination)
            move: Move | None = None
            print(f"\n[GAME {games_recorded + 1}] {args_cli.chess_scenario}: "
                  f"{args_cli.white} (white) vs {args_cli.black} (black)")

            while simulation_app.is_running():
                player = position.side_to_move
                arm = arms[player]

                if not pending and not arm.busy:
                    if move is not None:                       # previous move finished
                        position = position.apply(move)
                        plies += 1
                        move = None
                        knocked = toppled_pieces(inner, piece_names, position)
                        if knocked:
                            aborted = f"knocked over {', '.join(knocked)}"
                            print(f"  [ABORT] {aborted}")
                            break
                        outcome = position.result()
                        if outcome is not None or plies >= args_cli.max_plies:
                            results[outcome or "unfinished"] = results.get(outcome or "unfinished", 0) + 1
                            print(f"  result after {plies} plies: {outcome or 'ply cap reached'}")
                            break
                        continue

                    move = choose_move(position, rng)
                    mover = position.pieces[move.piece]
                    print(f"  ply {plies + 1:2d} {position.side_to_move:5s} "
                          f"{mover.kind:6s} {move}", flush=True)
                    if move.captured is not None:
                        slots = trays[player]
                        pending.append((player, move.captured, slots[min(plies, len(slots) - 1)]))
                    pending.append((player, move.piece, env_cfg.square_pos(layout, *move.target)))

                if pending and not arm.busy:
                    who, piece_index, destination = pending[0]
                    piece = inner.scene[piece_names[piece_index]]
                    others = [
                        i for i, other in enumerate(position.pieces)
                        if other.alive and i != piece_index
                    ]
                    neighbours = None
                    if others:
                        neighbours = (
                            torch.cat([inner.scene[piece_names[i]].data.root_pos_w.torch for i in others]),
                            torch.tensor(
                                [geometry[position.pieces[i].kind]["base_diameter"] / 2.0 for i in others],
                                device=inner.device,
                            ),
                            torch.tensor(
                                [TABLE_TOP_Z + geometry[position.pieces[i].kind]["height"] for i in others],
                                device=inner.device,
                            ),
                        )
                    ok = arms[who].plan(
                        piece.data.root_pos_w.torch, piece.data.root_quat_w.torch,
                        position.pieces[piece_index].kind, destination, neighbours,
                    )
                    if not ok:
                        aborted = (f"{who}/{piece_names[piece_index]}: {arms[who].fail_reason}")
                        print(f"  [ABORT] {aborted}")
                        break
                    pending.pop(0)

                commands, gripper_values = [], []
                for name in ("white", "black"):
                    command, gripper, _ = arms[name].step()
                    commands.append(command)
                    gripper_values.append(torch.full((1, 1), gripper, device=inner.device))
                action = torch.cat(
                    [commands[0], gripper_values[0], commands[1], gripper_values[1]], dim=-1
                )
                _, _, terminated, truncated, _ = env.step(action)
                if bool(terminated[0]) or bool(truncated[0]):
                    aborted = "episode timed out"
                    print("  [ABORT] episode clock expired")
                    break

            # Order matters and is not obvious: record_pre_reset *overwrites* the
            # success flag from the termination manager's "success" term, and this env
            # deliberately has none -- whether a game ended in checkmate is a rules fact,
            # not a scene one. Setting success first therefore silently loses it, and
            # EXPORT_SUCCEEDED_ONLY then writes an empty file while the log claims the
            # games were recorded. Close the episode first, then flag it, then export.
            inner.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            inner.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[aborted is None]], dtype=torch.bool, device=inner.device)
            )
            inner.recorder_manager.export_episodes([0])
            games_recorded += int(aborted is None)
            env.reset()
            for arm in arms.values():
                arm.phase = len(PHASES)

    print(f"\n[INFO] {games_recorded}/{args_cli.num_games} games recorded in {attempts} attempts")
    print(f"[INFO] outcomes: {results}")
    print(f"[INFO] Dataset: {dataset_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
