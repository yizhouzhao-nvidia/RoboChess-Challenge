# RoboChess on newton-physics

The [newton-physics](https://github.com/newton-physics/newton) port of the Isaac Lab task in
[lab/](../lab). Two of that task's capabilities are reproduced here:

1. **Visual inspection** — any of the five chess scenarios with any of the four supported arms.
2. **Chess picking** — pick-and-place of chess pieces, with grasps from
   [GraspGen](https://github.com/NVlabs/GraspGen).

Nothing outside `newton/` is modified; `lab/` stays the Isaac Lab reference and keeps working. The
board is not re-implemented either: `robochess_newton/board_layout.py` loads
`lab/source/robochess/tasks/manager_based/chess/board.py` through importlib, so both ports lay out
their squares from the same code. The grasps are the same
`assets/chess/generated/s150/grasps/chess_grasps.json`, the pieces are the same baked USDs, and the
pick controller is a port of `lab/scripts/generate_chess_pick_demos.py` down to its success
criterion.

What this port does *not* have: the Isaac Lab manager stack, `RecorderManager`, HDF5 datasets, RL
environments, teleoperation or XR. It is a `newton.ModelBuilder` scene, a `SolverMuJoCo` step, and a
scripted controller driven by `newton.ik`.

## Install

There is nothing to build and no package to install — both scripts put `newton/` on `sys.path`
themselves and defend against this repo's own `newton/` directory shadowing the `newton` package, so
they run from any working directory. What you need is one of the two supported interpreters and the
repo's LFS assets.

| | interpreter | newton | warp | mujoco |
|---|---|---|---|---|
| primary | `/home/yizhou/Projects/newton/.venv/bin/python` | 1.6.0.dev0 | 1.17 | 3.11 |
| secondary | `/home/yizhou/Projects/IsaacLab/env_isaaclab/bin/python` | 1.2.1 | 1.13 | 3.8 |

The two are not interchangeable at the library level (newton 1.6 needs warp >= 1.16), so pick one
and stay in it for a run. Every module here stays inside the API subset the two versions share, and
both entry points run on both, from any working directory. Where the two differ in behaviour rather
than in API — and they do, in speed and in one arm's dynamics — it is called out below. Unless a
command says otherwise, everything here was run on the primary interpreter.

Chess assets (piece USDs, boards, `pieces.json`, the GraspGen grasp JSON) are committed through
git-lfs:

```bash
git lfs pull
```

Robot assets are fetched on first use into `~/.cache/newton` (93 MB for the Franka, 86 MB for the
Rebot, 11 MB for the YAM) at git refs pinned by `robochess_newton/robots.py`, because the two
interpreters ship *different* default refs and would otherwise load different geometry. Franka and
Rebot come from [newton-assets](https://github.com/newton-physics/newton-assets), YAM from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), Piper from
`assets/piper/piper_camera.usd` in this repo. `ROBOCHESS_NEWTON_ASSETS_REF`,
`ROBOCHESS_MENAGERIE_REF` and `ROBOCHESS_MENAGERIE_DIR` override them.

GraspGen is **not** needed at runtime — its output is committed, and this port only reads it. The
baking and inference pipeline that produced it (`prepare_chess_assets.py`,
`graspgen_chess_grasps.py`, and the GraspGen venv they need) belongs to the Isaac Lab side and is
documented in [`lab/Readme.md`](../lab/Readme.md) and
[`lab/docs/franka_chess_picking.md`](../lab/docs/franka_chess_picking.md). Re-run it only if you
change the piece scale; `--grasp-file` points this port at the result.

The first `SolverMuJoCo` step on a cold `~/.cache/warp` spends about 49 s compiling kernels. That is
not a hang; subsequent runs take ~0.3 s.

Every command below is written from the repo root, which is the only thing the relative
`newton/scripts/...` path needs; give the script an absolute path and it runs from anywhere.

## Visual inspection

One arm, one scenario, nothing commanded: the arm holds its home posture on position drives while
the pieces settle onto the board. That is exactly the check this entry point exists for — the assets
load, the arm stands *on* the table and not through the board, and the pieces are where the layout
says they are.

```bash
PY=/home/yizhou/Projects/newton/.venv/bin/python

$PY newton/scripts/visualize_chess.py --robot franka --scenario 4x4 --viewer null --num-frames 120
```

```
  world 0: franka       4x4    board=board_4x4.usdc       scale=1.40 center=(+0.220,+0.000) pieces=16 bodies=0..29 ee=10 arm_dofs=0..6 reachable=16/16 targets=6
  total: worlds=1 bodies=30 shapes=332 dofs=105 contacts_max=32768 visual=True newton_worlds=1 device=cuda:0
  build=1228ms finalize=928ms substeps/frame=4 dt=0.00417
  120 frames in 1.61s (13.4 ms/frame, render included): dz_mm -1.67..-1.25 |qd|max=0.0008 nan=False contacts=23902
```

`dz_mm` is the summary line that matters: it is how far every piece moved vertically over the run.
Settled pieces end 1.25–1.67 mm below where they spawn (the 2 mm spawn clearance minus the convex
hulls' undershoot; the spread across that band is per piece kind, not error, and `1d` bottoms out at
−1.59 mm because of its particular mix of kinds). A `dz` near −300 mm would mean the contact budget
was undersized and every piece free-fell, which is the one silent failure mode of this scene. The
script exits non-zero only if a NaN appears.

## Robot options

The four supported arms are the four with a real parallel-jaw gripper, matching the Isaac Lab
picking task's `CHESS_ROBOTS`:

| `--robot` | source | arm + gripper DOFs | jaw span | reach | board distance |
|---|---|---|---|---|---|
| `franka` | newton-assets `franka_emika_panda/urdf/fr3_franka_hand.urdf` | 7 + 2 | 80 mm | 0.68 m | 0.45 m |
| `piper` | repo `assets/piper/piper_camera.usd` | 6 + 2 | 70 mm | 0.42 m | 0.30 m |
| `rebot` | newton-assets `seeed_rebot_devarm/urdf/seeed_rebot_devarm.urdf` | 6 + 2 | 90 mm | 0.44 m | 0.30 m |
| `yam` | Menagerie `i2rt_yam/yam.xml` | 6 + 2 | 79 mm | 0.42 m | 0.30 m |

```bash
$PY newton/scripts/visualize_chess.py --robot piper --scenario 4x4 --viewer null --num-frames 120
$PY newton/scripts/visualize_chess.py --robot rebot --scenario 4x4 --viewer null --num-frames 120
$PY newton/scripts/visualize_chess.py --robot yam   --scenario 4x4 --viewer null --num-frames 120
```

`so101`, `ur10` and `flexiv_rizon` are **out of scope**. They still resolve through
`robots.get_spec()` (they are listed in `ROBOTS` behind `UNSUPPORTED_ROBOTS`) but they are not a
`--robot` choice on either script, nothing is verified against them, and the last two have no
gripper at all.

Jaw span is the commanded opening; `robots.py` documents where it differs from the raw asset limit
and why (Piper's USD allows 50 mm per finger, YAM's finger *body* origins move opposite to its pads,
Rebot's fingers are wedge-shaped so the usable slot at the TCP is 71 mm rather than 90 mm). Reach
and board distance are carried over from the Isaac Lab `ChessRobotSpec` unchanged; the Rebot and YAM
*home postures* are not — they had to be re-solved, because the Newton sources are a URDF and an
MJCF with different joint frames from the USDs Isaac Lab loaded.

## Chess layouts

The same five scenarios as `lab/`, selected with `--scenario`:

```bash
# one of every piece kind on a compact 3x3 board, sized so the shorter arms reach all of it
$PY newton/scripts/visualize_chess.py --scenario pieces --viewer null --num-frames 120

# 1D: king, knight, rook vs rook, knight, king
$PY newton/scripts/visualize_chess.py --scenario 1d --viewer null --num-frames 120

# pawn-only 3x3
$PY newton/scripts/visualize_chess.py --scenario 3x3 --viewer null --num-frames 120

# 4x4 Mallett-Hill-Boyer minichess (default)
$PY newton/scripts/visualize_chess.py --scenario 4x4 --viewer null --num-frames 120

# standard 8x8 opening, 32 pieces
$PY newton/scripts/visualize_chess.py --scenario 8x8 --viewer null --num-frames 120
```

`--board-scale` stretches the squares; the default `auto` uses the per-scenario value from the lab
config — 1.4 everywhere except `8x8`, which stays at 1.0 because stretching a 0.48 m board by 1.4
pushes most of it outside the Franka's reach. At 1.0 the 1.5x pieces fill 78–98 % of a 60 mm square,
which is why every other scenario is widened: 1.4 opens ~25 mm of room beside a piece for a 22 mm
finger to descend into.

## Several worlds, several arms

`--world-count` replicates the scene. `--robot`, `--scenario` and `--board-scale` also take
comma-separated lists, which gives the worlds different content:

```bash
# four identical worlds, stamped out with ModelBuilder.replicate
$PY newton/scripts/visualize_chess.py --scenario 3x3 --world-count 4 --viewer null --num-frames 120

# a franka on 8x8 next to a yam on 1d
$PY newton/scripts/visualize_chess.py --robot franka,yam --scenario 8x8,1d --world-count 2 \
    --viewer null --num-frames 120
```

Identical worlds become real newton worlds. Worlds that differ cannot: `SolverMuJoCo` requires
homogeneous worlds, so mixed runs are laid side by side inside a *single* newton world at the
offsets `replicate` would have used. `describe()` prints `newton_worlds=1` when that happens.
Replicas agree with world 0 to 1.2e-4 mm after 120 frames.

## Rendering

With a display, `--viewer auto` opens an interactive OpenGL window. Without one — this box has no
`$DISPLAY` — it resolves to the USD viewer and writes a stage you can open elsewhere, unless you
asked for images or video, in which case it renders offscreen through EGL. `--viewer gl --headless`
forces that offscreen path explicitly.

```bash
# USD stage (works anywhere)
$PY newton/scripts/visualize_chess.py --robot franka --scenario 4x4 \
    --viewer usd --num-frames 60 --output-path out/chess_franka_4x4.usd

# offscreen OpenGL through EGL -> one PNG per frame
$PY newton/scripts/visualize_chess.py --robot franka --scenario 8x8 --viewer gl --headless \
    --save-images out/franka_8x8 --num-frames 180
```

MP4 needs `imageio-ffmpeg`, which only the Isaac Lab venv has:

```bash
/home/yizhou/Projects/IsaacLab/env_isaaclab/bin/python newton/scripts/visualize_chess.py \
    --robot franka --scenario 8x8 --viewer gl --headless \
    --save-video out/franka_8x8.mp4 --num-frames 180
```

Those three produce, respectively, a 22 MB stage with 60 timesamples, 180 PNGs (31 MB) named
`frame_00000.png` upward, and a 121 kB h.264 mp4. In the newton venv `--save-video` prints a warning
naming the missing backend and the directory it writes PNGs to instead — it does not fail, and it
does not silently produce nothing.

`--zoom`, `--camera=x,y,z` and `--camera-target=x,y,z` move the camera (the `=` form is required for
a negative coordinate, or argparse reads the leading `-` as an option). `--width`/`--height`/`--fps`
size the framebuffer and the timeline. `--viewer` also accepts `file`, `rerun`, `viser` and `null`.

## Chess picking

Each episode hands every world one move — pick piece *i* up, put it on destination *j* — and
`robochess_newton.pick.ChessPickTask` executes it with the GraspGen grasp the board actually leaves
room for. The nine-phase schedule (`pre_grasp, descend, close, lift, transfer, place, release,
retreat, settle`), the tolerances and the success criterion are ported from the Isaac Lab policy:
the piece must end within 20 mm of its destination square, within 10 mm of the surface, upright to
25 degrees, at rest, and out of the fingers.

```bash
$PY newton/scripts/run_chess_pick.py --robot franka --scenario 4x4 \
    --world-count 8 --num-episodes 16 --seed 0 --viewer null
```

```
[episode   4] world 3 king   piece_white_king_0 -> (+0.255,-1.013)  success         phase=release   steps=288 place_err=   1.4mm grasp=0.941/pen=0.0mm
...
[INFO] 16 successes / 16 attempts (100%) in 41.3s (582 control ticks, 1164 frames)
[INFO] by piece kind: king=3/3, knight=4/4, pawn=9/9
```

An episode ends as `success`, or with the reason it did not. The outcome is latched as soon as the
schedule finishes, so a failure names the clause of the success predicate that rejected it —
`missed_target`, `off_surface`, `piece_tipped`, `not_released`, `still_moving` — or the event that
ended it early: `board_disturbed` (a piece that was not the target got knocked over),
`piece_off_board` (the piece left the table), `timed_out` (the step budget ran out mid-schedule,
which does not happen at the shipped budget). The line also reports the phase it ended in, the
placement error and the grasp quality. All worlds reset together, so a batch of N worlds runs N
episodes at a time.

`--balance-kinds` steers each episode toward the least-attempted piece kind, so a short run still
covers pawn, rook, knight, bishop, queen and king. `--debug` traces world 0 through every control
tick. `--seed` picks the move sequence, and the same seed reproduces the same run.

Record one episode (Isaac Lab venv, for the mp4):

```bash
/home/yizhou/Projects/IsaacLab/env_isaaclab/bin/python newton/scripts/run_chess_pick.py \
    --robot franka --scenario 4x4 --world-count 1 --num-episodes 1 --seed 0 --zoom 2.2 \
    --viewer gl --headless --save-video out/franka_4x4_pick.mp4 --save-images out/franka_4x4_pick
```

At this seed that is 648 frames (a 5.0 MB mp4) of the arm lifting the black knight off the board
and setting it on the capture tray:

```
[episode   1] world 0 knight piece_black_knight_0 -> (+0.185,-0.433)  success  phase=release   steps=324 place_err=   2.6mm grasp=0.977/pen=1.5mm
```

### Support matrix

Every cell below was run as

```bash
$PY newton/scripts/run_chess_pick.py --robot <arm> --scenario <scenario> \
    --world-count 8 --num-episodes 16 --seed 0 --viewer null
```

with `--world-count 6 --num-episodes 12` on `8x8`, everything else at its CLI default, on the
primary interpreter. Visualization works in all twenty cells; the number is the **pick success
rate**.

| arm | `pieces` | `1d` | `3x3` | `4x4` | `8x8` |
|---|---|---|---|---|---|
| `franka` | **16/16** 100% | **16/16** 100% | **16/16** 100% | **16/16** 100% | 9/12 75% |
| `piper` | 14/16 88% | 14/16 88% | **16/16** 100% | 15/16 94% | 10/12 83% |
| `rebot` | 6/16 38% | 8/16 50% | 15/16 94% | 11-14/16 (median 13) | 1-3/12 |
| `yam` | 12/16 75% | **16/16** 100% | **16/16** 100% | 12/16 75% | 9/12 75% |

### Notes on the matrix

* **These are 16-sample estimates, and the weaker cells are not reproducible run to run.**
  Fixing the seed fixes the *move sequence*, not the trajectory: `SolverMuJoCo` on GPU is not
  bitwise deterministic across runs, and where a grasp is marginal that difference decides the
  episode. `rebot`/`4x4` spans 11-14/16 over ten identical invocations at seed 0, and three
  identical runs differ in placement error from episode 1 onward. The Franka is stable in outcome,
  step count and tick count (`4x4` is 16/16 at 582 control ticks every time) and varies only in the
  0.1 mm digit. Read a single cell as +-1 for the strong arms and +-3 for `rebot`.
* **`3x3` is the floor of the port** — pawns only, comfortably inside every arm's reach, and three
  of the four arms take it 16/16.
* **The Franka is the reference arm.** It is the only one at 100 % on three boards, and its 0.68 m
  reach is the only one that gets near a full 8x8 board (30 of 32 pieces, 36 legal destinations)
  or leaves more than one legal destination on `4x4` — see the `targets=1` limitation below.
* **The Rebot is the weak arm**, and its failures are almost all `missed_target` — 9 of 16 on
  `pieces`, 8 of 16 on `1d`. `--debug` shows why: it carries a pawn at roughly 38 mm of jaw opening, i.e.
  around the piece's flare rather than its 11 mm neck, so the piece swings during the place descent
  and drifts past the 20 mm tolerance. Its `tcp_offset` is the suspect. On the low, well-spaced
  pieces of `3x3` and `4x4` it is fine (94 % and 88 %).
* **`8x8` is the hardest board for every arm**, for the reason you would expect: it is the one
  scenario that does not get the 1.4x board stretch, so it is 60 mm squares against an 80–90 mm open
  hand. The Franka's three failures there are all `board_disturbed` during `descend` — on a knight,
  a knight and a bishop — the open fingers clipping a neighbour on the way in.
* **Per-kind counts are small-n.** The seed draws whatever it draws, so a cell's `rook=0/1` is one
  episode, not a verdict on rooks. `--balance-kinds` is the flag for probing a kind on purpose, and
  it makes the task harder rather than easier: `yam --scenario 4x4 --balance-kinds` returns 10/16
  (62 %) with rook 0/2 and king 0/1 where the natural draw gives 12/16. The rook is the recurring
  weak kind for the three shorter arms, and the tall pieces are what the crowded boards lose.
* **`franka --scenario pieces` is the strongest single result**: 16/16 with bishop 3/3, king 3/3,
  pawn 5/5, queen 4/4, rook 1/1 — one of every kind, so the 7–9 mm necks of the tall pieces are all
  being pinched successfully. Adding `--balance-kinds` to the same command is also 16/16, with the
  draws spread as bishop 2/2, king 2/2, knight 1/1, pawn 5/5, queen 4/4, rook 2/2.
* **`--num-grasp-candidates 64` buys back most of the `8x8` gap.** The grasp JSON holds 64
  candidates per piece; the default of 12 is there for parity with the Isaac Lab script. Raising it
  helps the crowded boards, and specifically the knight — the one kind that is not a solid of
  revolution, so it gets no yaw spin with which to thread the fingers between its neighbours:

  ```bash
  $PY newton/scripts/run_chess_pick.py --robot franka --scenario 8x8 \
      --world-count 6 --num-episodes 12 --seed 0 --num-grasp-candidates 64 --viewer null
  ```

  `8x8` 9/12 -> **11/12**, with knight 0/2 -> 2/2; `1d` 14/16 -> **16/16**. It costs about 38 ms of
  planning per world per episode instead of 8.
* **Widening the board does not fix `8x8`.** `--board-scale 1.3` on the same 12 episodes is still
  9/12, the failures simply move to the tall pieces at the edge of reach (bishop 0/1, queen 0/1),
  and the episodes run far longer than the unstretched board's (against 606 control ticks).
* **The two interpreters do not agree for every arm.** Identical source, identical command,
  identical seed, run again under newton 1.2.1 in the Isaac Lab venv:

  | case | newton 1.6 | newton 1.2.1 |
  |---|---|---|
  | `franka` `4x4` | 16/16 100% | 16/16 100% |
  | `yam` `4x4` | 12/16 75% | 14/16 88% |
  | `rebot` `4x4` | 14/16 88% | **1/16 6%** |

  The Franka is not merely equal but identical — the same 582 control ticks on both. The Rebot is
  the opposite: at the shipped drive gains it is effectively broken on 1.2.1, and `--arm-ke 2000`
  (the value that was reported to rescue it) only recovers it to 6/16. That is a solver-version
  difference, not a code difference, and no per-version gain is baked in.

## Where this maps onto `lab/`

| Isaac Lab | Newton |
|---|---|
| `zero_agent.py --task RoboChess-Visual-v0` | `newton/scripts/visualize_chess.py` |
| `generate_chess_pick_demos.py` | `newton/scripts/run_chess_pick.py` |
| `chess/board.py` | reused as-is, loaded through importlib |
| `robot_configs.py` `ChessRobotSpec` / `CHESS_ROBOTS` | `robots.py` `NewtonRobotSpec` / `ROBOTS` |
| `FrankaChessSceneCfg` + `ChessPickEnvCfg` | `scene.py` `ChessScene` / `ChessWorld` |
| `ChessPickPolicy` | `pick.py` `ChessPickTask` |
| differential-IK action term, robot root frame | `newton.ik` Levenberg-Marquardt, world frame |
| PhysX articulation + rigid bodies | `SolverMuJoCo` + convex-hull piece colliders |
| `RecorderManager` -> HDF5 | not ported; outcomes are printed |

## Deliberate differences from the Isaac Lab task

* **The board is a collider here.** Isaac Lab's chessboard is an `AssetBaseCfg` with no collision
  API and its pieces rest on the *table*; the shipped board USDs are zero-thickness quads, which no
  physics engine can rest a piece on, so this port adds a thin static box whose top face is exactly
  at the table top. The contact plane is identical; the friction the piece sees is 0.9 rather than
  the table's 1.0.
* **One friction number per surface.** Newton takes a single `mu` where Isaac Lab has static and
  dynamic. Pieces use the static value (1.1), the table and board the dynamic one (0.9).
* **Rebot and YAM home postures were re-solved** for the URDF/MJCF sources (see above).
* **Two departures in the pick schedule**, both because `newton.ik` tracks its command far more
  tightly than the differential-IK action term the Isaac Lab schedule was tuned against: a leg's
  last commanded pose is the goal rather than `(N-1)/N` of the way to it, and `transfer` ends at the
  carry height rather than a fixed 120 mm so the carry is level instead of a descending sweep across
  the pieces still standing. Both are documented with their measured effect in the `pick.py` module
  docstring.
* **The pieces get a real inertia tensor.** Building the hulls at `density=0` (necessary, or the
  overlapping hulls inflate the mass by 16 %) leaves `body_inertia` at zero and Newton substitutes
  an isotropic fallback, which made every piece ~10x too easy to topple. `PieceAssets` integrates
  the hulls instead and reproduces `add_usd`'s own inertia to float32.
* **Three asset defects are worked around in code**, each of which was a silent wrong answer:
  YAM's `mujoco:gravcomp` keys are stale after `collapse_fixed_joints` and cancelled gravity on the
  first five chess pieces; `mujoco:actuator_trnid` is not offset by `add_builder`, so a merged
  second arm's torque limits clamped the *first* arm; and the Rebot URDF points its collision tags
  at full-resolution STLs (364 k vertices), which is both slow and unstable until they are
  approximated by convex hulls. `ChessScene.describe()` reports the first two when they fire.
* **YAM's imported actuator force ranges are cleared.** newton 1.6 honours the MJCF's
  `forcerange`/`ctrlrange` and 1.2.1 ignores them, so left alone the same source gives the YAM
  genuinely different dynamics on the two interpreters — torque-limited on one, unlimited on the
  other, with the gravity compensation that is routed through those actuators clipped away.
  `robots.py` clears the rows so 1.6 agrees with 1.2.1.

## Package layout

```
newton/robochess_newton/
  board_layout.py   importlib shim onto lab's board.py + the table/board/tray constants
  assets.py         PieceAssets hull+mesh cache, board and table builders, contact budget
  robots.py         NewtonRobotSpec table, per-arm loaders, GraspGen retargeting math
  viewer_utils.py   viewer selection and CLI flags, EGL headless PNG/MP4 capture
  scene.py          ChessScene / ChessWorld — the assembled scene and its index bookkeeping
  grasps.py         GraspGen JSON -> per-arm end-effector poses, re-scored against the board
  pick.py           the nine-phase IK-driven pick-and-place state machine
newton/scripts/
  visualize_chess.py   capability 1
  run_chess_pick.py    capability 2
newton/example_so101.py   pre-existing standalone SO-101 example, not part of this port
```

Both scripts have detailed `--help`. `newton/example_so101.py` predates this work and is unrelated
to it; it currently does not run (a stray `ipdb.set_trace()` at line 66).

The package is usable without the scripts. Put `newton/` on `sys.path` and build a scene:

```python
import sys
sys.path.insert(0, "newton")  # or the absolute path to <repo>/newton

from robochess_newton.scene import ChessScene

scene = ChessScene(robot="franka", scenario="4x4")
scene.finalize()          # builds the Model, the SolverMuJoCo and the states
scene.apply_home_targets()
for _ in range(120):
    scene.step()          # one frame = 4 physics substeps at 1/240 s

world = scene.worlds[0]
print(world.piece_names[0], world.square_position(0, 0))
print(len(world.reachable_piece_indices()), "reachable pieces,", len(world.target_positions()), "destinations")
```

```
piece_white_king_0 (0.094, -0.126, 0.77)
16 reachable pieces, 6 destinations
```

`ChessWorld` is the index bookkeeping a flat `newton.Model` does not give you: which body is which
piece, which coordinates are the arm, where each square is in world space. `ChessPickTask` from
`pick.py` drives it; `run_chess_pick.py` is a thin wrapper around the two.

## Limitations

Found by the verification passes and **not** fixed. None of them crashes or hangs; they are honest
weaknesses of the port. (The defects that *were* fixed -- `--num-frames` corrupting the statistics,
episodes labelled `timed_out` after finishing their schedule, the CLI validation gaps, the stale
armature measurement, YAM's TCP living in the wrong module -- are gone; a re-verification pass
confirmed the fixes moved labels and tick counts but no success count.)

**Results you can misread**

* **Piper, Rebot and YAM have exactly one legal destination on `4x4`.** `targets=1` in the
  `describe()` line: their 0.42-0.44 m reach plus the table-edge filter leaves a single square, so
  every episode for those three arms places on the same square. Franka gets 6. This is inherited
  from the lab's own target filter, not introduced here, but it means a long run on those arms tests
  one destination.
* **The weaker cells of the matrix move between identical runs** -- see the note under the matrix.
  `rebot`/`4x4` spans 11-14/16 at a fixed seed.
* **`timed_out` is now rare rather than informative.** Since episodes latch their outcome when the
  schedule completes, `timed_out` means the step budget ran out *mid-schedule*; it did not occur
  once in 552 episodes at the shipped budget. A failing episode reports which clause of the success
  predicate rejected it: `missed_target`, `off_surface`, `piece_tipped`, `not_released`,
  `still_moving`, plus `board_disturbed` and `piece_off_board`.

**Physics and rendering**

* **The Rebot picks reliably only on newton 1.6.** On 1.2.1 the same command drops from ~13/16 to
  1/16 (see the cross-interpreter table above). Pose tracking blows out to 44 mm / 13.8 deg during
  `lift` where 1.6 holds a few mm -- same source, same gains, so it is a solver-version difference.
  `--arm-ke 2000` recovers it to about 6/16. No per-version gain is baked in. The other three arms
  are fine on both.
* **`rebot` on `8x8` does not settle like the other arms**: repeatably one piece rests ~0.8 mm high
  (`dz_mm -1.72..-0.55` against everything else's -1.67..-1.25 band), which looks like a piece
  resting on the arm's base plate.
* **The Rebot grips high on the piece.** `--debug` shows it carrying a pawn at ~38 mm of jaw
  opening -- around the flare rather than the 11 mm neck -- so the piece swings during the place
  descent. Its `tcp_offset` is the suspect, and it is the likeliest single fix for the whole rebot
  column.
* **`8x8` is the genuine weak board for every arm.** It is the one scenario that does not get the
  1.4x board stretch, so it is 60 mm squares against an 80-90 mm open hand.
  `--num-grasp-candidates 64` recovers most of the gap; widening the board does not.
* **One `SolverMuJoCo` per process.** Two in one process segfaults, so anything that rebuilds a
  scene has to fork. Both scripts build exactly one.

A previously documented cosmetic defect -- the YAM gripper rendering as saturated
magenta/cyan/yellow collision primitives -- **did not reproduce** on re-check: the same command
plus a close-up render both show clean grey/white jaws. It is recorded here only so the next person
to see it knows it has been looked for twice.

## Notes for maintainers

Hard-won facts about this stack, each of which cost a debugging cycle to find. They are load-bearing:
several are the reason a line of code looks the way it does.

* **`PXR_WORK_THREAD_LIMIT=1` is mandatory.** `pxr.UsdPhysics.LoadUsdPhysicsFromRange()`, which
  `ModelBuilder.add_usd()` calls, segfaults nondeterministically when it is allowed a thread pool --
  a bare loop over one piece USD died after 5-23 iterations, three runs out of three.
  `board_layout.py` sets it at import and is the module every other one imports first, so no import
  path can skip it.
* **Piece colliders must be `add_shape_convex_hull`, never `add_shape_mesh`.** Measured on 4x4 with
  identical geometry: 5.16 ms/frame and stable, against 145.76 ms/frame and the simulation
  diverging. The same lesson applies to imported robots -- the Rebot URDF points its collision tags
  at 364 k-vertex STLs, which `robots.py` convex-hulls on import (15.51 -> 3.11 ms/step).
* **The contact budget is a silent failure mode.** Undersize `rigid_contact_max` and the pipeline
  drops *every* contact: each piece free-falls, `dz` reads -311.67 mm, and nothing warns you.
  `assets.contact_budget()` sizes it at 1600 contacts per piece rounded to a power of two.
* **The board has to be a thin static box, not its mesh.** The shipped board USDs are zero-thickness
  quads. As a mesh collider pieces fall through (`dz` -1182 mm); as a convex hull they half fall
  through (-492 mm); as a box they rest (-1.7 mm).
* **`use_mujoco_contacts=True` is broken with `replicate()`** -- only world 0 collides.
* **`SolverMuJoCo` refuses heterogeneous worlds**, which is why mixed runs are laid side by side
  inside a single newton world.
* **Two asset defects are worked around at load time**, both silent: YAM's `mujoco:gravcomp` keys
  are stale after `collapse_fixed_joints` and cancelled gravity on the first five chess pieces, and
  `mujoco:actuator_trnid` is not offset by `add_builder`, so a merged second arm's torque limits
  clamped the *first* arm. `ChessScene.describe()` reports both when they fire.
* **`ViewerUSD` drops the colours it is given, and we shim it.** Its `log_instances()` does
  `PrimvarsAPI(instance).GetPrimvar("displayColor").Set(...)` on a prim that never had the primvar
  created, so the `Set` is a silent no-op: every mesh in the exported stage renders in the default
  grey, and the black chess pieces come out white in usdview and Isaac Sim. The GL viewer has its
  own path and was always correct, which is what makes this easy to miss.
  `viewer_utils._fix_instance_colors()` wraps the method and creates the primvar first. Present in
  both newton 1.2.1 and 1.6.0.dev0; drop the shim if upstream fixes it.
* **Never run the Isaac Lab interpreter with cwd `~/Projects/newton`** -- it imports the 1.6 source
  tree while `newton.__version__` still reports 1.2.1 from the installed distribution's metadata.
* **A cold `~/.cache/warp` costs ~49 s** on the first `SolverMuJoCo.step` while kernels compile,
  ~0.3 s warm. It is not a hang.
* **The repo's own `newton/` directory is a latent shadowing hazard.** Today it has no
  `__init__.py`, so PEP 420 makes it a namespace *portion* and the installed `newton` still wins
  (measured on both versions). Add an `__init__.py` and it would beat site-packages outright.
  `board_layout.shield_newton_imports()` drops repo-root entries from `sys.path` to close that door.

**Speed**

* **newton 1.2.1 is much slower on the crowded boards**, because 1.6's narrow phase is faster.
  `visualize_chess.py --robot franka --viewer null --num-frames 120`, same machine, uncontended:

  | scenario | newton 1.6 | newton 1.2.1 |
  |---|---|---|
  | `4x4` | 13.4 ms/frame | 24.3 ms/frame |
  | `8x8` | 19.4 ms/frame | 107.8 ms/frame |

  Pick runs scale the same way. Prefer the primary interpreter for everything except mp4 output.
