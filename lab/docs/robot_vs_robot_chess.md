# Robot vs robot: two arms playing chess

Two robot arms face each other across a board, one playing white and one playing black.
Moves come from a rule set rather than being sampled, so the arms play a real game:
legal moves only, captures cleared off the board, and a result at the end — checkmate,
stalemate, or a Hexapawn win.

This is a different task from the single-arm [pick-and-place](franka_chess_picking.md).
That one moves a randomly chosen piece to a randomly chosen square; this one plays chess.

```bash
LAB=/home/yizhou/Projects/IsaacLab/env_isaaclab/bin/python   # Isaac Lab 3.0 / Isaac Sim 6.0
```

---

## Quick start

```bash
# Check the rules first -- no simulator needed, runs in about a second
python3 lab/scripts/test_chess_rules.py

# Play three games of 3x3 Hexapawn, two Frankas
$LAB lab/scripts/play_chess_game.py --headless \
     --chess_scenario 3x3 --white franka --black franka \
     --num_games 3 --max_plies 8 --policy random \
     --dataset_file ./lab/datasets/chess_game_hexapawn.hdf5
```

Watch it instead of recording it by dropping `--headless`, or add `--video` to write an
mp4 alongside the dataset.

---

## The three variants

| `--chess_scenario` | Board | Pieces per side | Won by |
|---|---|---|---|
| `1d` | 1 x 8 cells | King, knight, rook | checkmate |
| `3x3` | 3 x 3 | Three pawns (Hexapawn) | reaching the far rank, capturing everything, or leaving the opponent with no move |
| `minichess` | 4 x 4 | K N N R + four pawns | checkmate |

**1D chess.** White `K N R` on cells 0-2 faces black `R N K` on cells 5-7. The king steps
one cell, the rook slides, and the knight jumps *exactly two cells over whatever is in
between*. No pawns, so no promotion. Draws by stalemate, threefold repetition, or bare
kings.

**Hexapawn.** Pawns step one square forward or capture one square diagonally forward.
No double step, no en passant. Note the win condition is the opposite of chess: a player
with no legal move *loses*.

**Minichess.** Ordinary chess movement clipped to a 4 x 4 board, with the back rank
mirrored (`K N N R` against `R N N K`) so the kings do not share a file. Pawns take a
single step and promote to a queen on the far rank. No castling, no en passant.

> `minichess` is a separate scenario from `4x4` on purpose. The `4x4` layout starts both
> kings on file 0, which is not a game — but a recorded dataset maps its piece indices
> through that layout, so it was left alone rather than fixed in place.

---

## Choosing the arms

Any pairing of the four supported arms works; they need not match.

```bash
$LAB lab/scripts/play_chess_game.py --headless --white franka --black piper ...
```

Reach is the constraint, and it is checked before the first move rather than discovered
as a stalled game:

```
[WARN] black (piper) cannot reach 4 square(s): [(0, 3), (1, 3), (2, 3), (3, 3)]
[WARN] black cannot reach 2 of its own capture-tray slots
```

Measured coverage, board squares only:

| | franka / franka | franka / piper | rebot / yam |
|---|---|---|---|
| `1d` | all reachable | black misses 2 | black misses 2 |
| `3x3` | all reachable | all reachable | all reachable |
| `minichess` | all reachable | black misses 4 | 2 and 4 missed |

**Two Frankas is the only pairing that reaches every square of every variant.** The short
arms are fine on Hexapawn.

---

## How a move is executed

The side to move plays; the other arm holds station. A capture is **two** pick-and-places:

1. lift the captured piece off its square and drop it in the capturing player's tray;
2. move the capturing piece onto the now-empty square.

Doing it in one step would drop a piece onto an occupied square and knock both over.

Each pick-and-place runs eight phases — `pre_grasp`, `descend`, `close`, `lift`,
`transfer`, `place`, `release`, `retreat` — advancing when the arm has *arrived*, with a
deadline so a bad IK solution cannot stall the game.

Each player has **its own capture tray**, pulled back towards its own base. One shared
tray does not work: on the 1D board it lands 0.70 m from a 0.68 m arm and every capture
fails.

---

## Options

| Flag | Default | Notes |
|---|---|---|
| `--chess_scenario` | `3x3` | `1d`, `3x3`, `minichess` |
| `--white`, `--black` | `franka` | `franka`, `piper`, `rebot`, `yam` |
| `--num_games` | 3 | Games to record. Aborted games are retried, up to `4 x num_games` attempts |
| `--max_plies` | 12 | Ply cap; a game that hits it is recorded as unfinished |
| `--policy` | `greedy` | `greedy` prefers mate then captures; `random` picks uniformly |
| `--dataset_file` | `./lab/datasets/chess_game.hdf5` | Output |
| `--seed` | 0 | Move choice only |
| `--debug` | off | Print every phase transition and its tracking error |
| `--video` | off | Write an mp4 next to the dataset |

**Prefer `--policy random` for recording.** Greedy is correct but plays for mate, and
minichess has a forced mate in one from the opening position — white's knight to `(1,2)`
checks the king on `(3,3)`, whose three escape squares are all blocked by its own pieces,
and a knight check cannot be blocked. Every greedy minichess game is therefore one ply
long.

### Episode clock

```
episode_length_s = max_plies x seconds_per_ply x 2
                 = max_plies x 26 s x 2
```

The factor of two is because a capture is two pick-and-places. Sizing for the worst case
matters: a game that runs out of clock is discarded whole, however many plies it had
already played correctly.

---

## What gets recorded

One game is one episode, written through Isaac Lab's `RecorderManager` with
`EXPORT_SUCCEEDED_ONLY`.

- **Actions**, 16 wide: a 7-DoF absolute pose plus a binary gripper, for each arm.
- **Observations**: `joint_pos`/`joint_vel`/`eef_pos`/`eef_quat`/`gripper` per player, plus
  `piece_positions` and `piece_orientations` for every piece on the board.

Two things about success are worth knowing if you extend this:

- Success is **not** a termination term. Whether a game ended in checkmate is a fact about
  the rules, not about where the pieces are, so the driver flags it through
  `set_success_to_episodes`.
- The call order is load-bearing. `record_pre_reset` re-derives success from the
  termination manager's `success` term and **overwrites** whatever you set. This env has
  no such term, so flagging success first silently stamps every episode `False` and
  `EXPORT_SUCCEEDED_ONLY` writes an empty file *while the log reports the games as
  recorded*. Close the episode, then flag it, then export.

### Checking a dataset

Read the file, not the driver's log:

```bash
$LAB -c "
import h5py
with h5py.File('lab/datasets/chess_game_hexapawn.hdf5') as h:
    print(len(h['data']), 'games',
          sum(int(h['data'][n].attrs['num_samples']) for n in h['data']), 'transitions')"
```

---

## Rules testing

The rules decide which piece the arm is told to pick up, so an error there is invisible in
simulation — the robot executes an illegal move perfectly. `test_chess_rules.py` runs
without Isaac Sim, under a bare interpreter:

```bash
python3 lab/scripts/test_chess_rules.py
```

```
verified 325 checkmates
1d         over 300 random games: black=55, draw=197, white=48
3x3        over 300 random games: black=131, white=169
minichess  over 300 random games: black=73, draw=102, white=125
All 10 rule checks passed.
```

It checks what a demo cannot show you: that no legal move leaves your own king in check,
that captured pieces vanish and free their square, that two pieces never share a square,
that knights jump, that pawns cannot capture forwards, that promotion works, and that
every declared checkmate really is one.

---

## Current status, and a known limitation

Recording yield, two Frankas, `--policy random`, 8-ply cap, 12 attempts each:

| Variant | Clean games recorded |
|---|---|
| `3x3` Hexapawn | **2-3 of 3** |
| `1d` | **0 of 3** |
| `minichess` | **0 of 3** |

Shipped: `lab/datasets/chess_game_hexapawn.hdf5`, 2 games, 5,898 transitions, verified from
the file — no toppled pieces left standing on the board, correct piece counts, captures in
the tray.

**Why 1D and minichess produce nothing.** A game is rejected if any piece still in play is
knocked over — more than 30 degrees from vertical, checked after every ply:

```
[ABORT] knocked over piece_white_knight_0, piece_white_rook_0
```

Without that check the driver reports every game a success while a quarter of the pieces
lie flat, which is worse than useless for imitation: it teaches a policy to plough through
the board.

The pieces knocked over are consistently **back-rank pieces nearest a robot's own base**,
rarely ones in the middle. That points at the transfer path rather than the grasp: each
phase interpolates in a straight line in task space from wherever the hand is to the next
waypoint, so after a capture the hand is out at the tray and the straight line back to the
next pre-grasp sweeps low across the near rank.

The fix is a via-point — rise to carry height, traverse, then descend — the same
waypointed transfer named as the top open item in the
[CoRL report](RoboChess_CoRL2026_Report.pdf). It is **not implemented**. Filtering grasps
for finger clearance against neighbouring pieces was tried and did not help: only 1 abort
in 36 cited it.

Hexapawn survives because its pieces are short and well spread.

---

## Adding a variant

1. Add the starting position to `make_layout` in `board.py` and list it in
   `PLAYABLE_SCENARIOS`.
2. Add movement and win conditions to `chess_rules.py`, and map the scenario in
   `scenario_variant`.
3. Add cases to `test_chess_rules.py` — at minimum the opening position, one movement rule
   that is easy to get wrong, and that random games terminate.
4. Give it an entry in `DEFAULT_BOARD_SCALE` in `franka_chess_env_cfg.py`.

Only step 3 needs a GPU-free minute; do it before touching the simulator.
