# RoboChess on genesis-world

The [genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) port of the Isaac Lab task in
[lab/](../lab). The same two capabilities the [newton-physics port](../newton) reproduces are reproduced
here:

1. **Visual inspection** — any of the five chess scenarios with any of the four supported arms.
2. **Chess picking** — pick-and-place of chess pieces, with grasps from
   [GraspGen](https://github.com/NVlabs/GraspGen).

Nothing outside `genesis-world/` is modified; `lab/` stays the Isaac Lab reference and `newton/` stays the
Newton reference, and both keep working. The board is not re-implemented either:
`robochess_genesis/board_layout.py` loads
`lab/source/robochess/tasks/manager_based/chess/board.py` through importlib, exactly as the Newton port
does, so all three ports lay out their squares from the same code. The grasps are the same
`assets/chess/generated/s150/grasps/chess_grasps.json`, the pieces are the same baked USDs, the robots are
the same files at the same pinned git refs, and the pick controller is a translation of
`newton/robochess_newton/pick.py` down to its success criterion.

What this port does *not* have: the Isaac Lab manager stack, `RecorderManager`, HDF5 datasets, RL
environments, teleoperation or XR. It is a `genesis.Scene`, a batched rigid solver step, and a scripted
controller driven by `RigidEntity.inverse_kinematics`.

## Install

There is nothing to build and no package to install — both scripts put `genesis-world/` on `sys.path`
themselves, so they run from any working directory. What you need is a Genesis interpreter and the repo's
LFS assets.

| | interpreter | genesis | torch | python |
|---|---|---|---|---|
| this port | `/home/yizhou/Projects/genesis/.venv/bin/python` | 1.3.3 | 2.13.0+cu132 | 3.12 |

Genesis is not in either of the Newton port's two venvs and the Newton stack is not in this one; they are
separate interpreters and nothing here is shared with them at the library level. To recreate it:

```bash
uv venv --python 3.12 /home/yizhou/Projects/genesis/.venv
uv pip install --python /home/yizhou/Projects/genesis/.venv/bin/python torch --torch-backend=auto
uv pip install --python /home/yizhou/Projects/genesis/.venv/bin/python genesis-world usd-core trimesh imageio imageio-ffmpeg
```

`usd-core` and `trimesh` are what the piece transcoder needs (Genesis pulls in `trimesh` itself, but not
`pxr`); `imageio`+`imageio-ffmpeg` are only for `--save-video`. Genesis 1.3.3 requires Python 3.10-3.13.

Chess assets (piece USDs, boards, `pieces.json`, the GraspGen grasp JSON) are committed through git-lfs:

```bash
git lfs pull
```

Robot assets are fetched on first use at git refs pinned by `robochess_genesis/robots.py` — the *same*
refs the Newton port pins, so both ports load byte-identical geometry. If `~/.cache/newton` already holds
a checkout at that ref (i.e. you have run the Newton port on this machine), it is reused rather than
downloaded again; otherwise a blobless sparse checkout lands in `~/.cache/robochess-genesis/assets`.
Franka and Rebot come from [newton-assets](https://github.com/newton-physics/newton-assets), YAM from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), Piper from
`assets/piper/piper_camera.usd` in this repo. `ROBOCHESS_NEWTON_ASSETS_REF`, `ROBOCHESS_MENAGERIE_REF` and
`ROBOCHESS_MENAGERIE_DIR` override them.

GraspGen is **not** needed at runtime — its output is committed, and this port only reads it. The baking
and inference pipeline that produced it belongs to the Isaac Lab side and is documented in
[`lab/Readme.md`](../lab/Readme.md) and
[`lab/docs/franka_chess_picking.md`](../lab/docs/franka_chess_picking.md). `--grasp-file` points this port
at a different result.

**One thing this port has that the other two do not: a transcode cache.** Genesis loads geometry from
files and has no in-memory mesh handle, so the baked piece USDs are converted once into
`~/.cache/robochess-genesis/pieces/s150/<kind>/` — 16 collision-hull OBJs, one decimated visual OBJ and a
one-link URDF that binds them to the authored mass and the computed inertia tensor. The boards get the
same treatment, one OBJ per square colour per board scale. It costs ~20 s on the first run and a handful
of `stat` calls afterwards; `$ROBOCHESS_GENESIS_CACHE` moves it. The cache is keyed on the source files'
size and mtime plus the transcoding options, so re-baking the assets or changing `--visual-faces`
regenerates rather than silently serving the old geometry.

Two noises you will see and can ignore: `PXR_WORK_THREAD_LIMIT is overridden to '1'` (this port sets it
before the first `pxr` import — openusd's parser segfaults nondeterministically when threaded, which is a
lesson inherited from the Newton port), and a wall of `profiling:/home/runner/... Cannot create directory`
on **stderr**, which is coverage instrumentation left in the `pygel3d` wheel Genesis depends on. Append
`2>/dev/null` if it is in the way.

Every command below is written from the repo root; give the script an absolute path and it runs from
anywhere.

## Visual inspection

One arm, one scenario, nothing commanded: the arm holds its home posture on position drives while the
pieces settle onto the board. That is exactly the check this entry point exists for — the assets load, the
arm stands *on* the table and not through the board, and the pieces are where the layout says they are.

```bash
PY=/home/yizhou/Projects/genesis/.venv/bin/python

$PY genesis-world/scripts/visualize_chess.py --robot franka --scenario 4x4 --viewer null --num-frames 120
```

```
  world 0: franka       4x4    board=board_4x4.usdc       scale=1.40 center=(+0.220,+0.000) pieces=16 env=0 ee_link=9 arm_dofs=0..6 reachable=16/16 targets=6
  total: worlds=1 genesis_envs=1 entities=23 max_collision_pairs=4096 visual=True links=34 geoms=276 dofs=105 backend=2
  render: none (no camera, no viewer)
  build=4621ms finalize=3072ms substeps/frame=4 dt=0.00417
  120 frames in 14.14s (117.8 ms/frame, render included): dz_mm -1.65..-1.24 |v|max=0.0000 nan=False
```

`dz_mm` is the summary line that matters: it is how far every piece moved vertically over the run. Settled
pieces end 1.24–1.65 mm below where they spawn (the 2 mm spawn clearance minus the convex hulls'
undershoot). **That band is the Newton port's, to 0.02 mm** — Newton reports −1.67..−1.25 on the same
scene, which is the strongest single check that the two ports are simulating the same geometry with the
same mass properties. A `dz` near −300 mm would mean the pieces free-fell. The script exits non-zero only
if a NaN appears.

Measured over all twenty arm x scenario cells, **nineteen** land in that band (`1d` reads −1.58..−1.24 and
`3x3` −1.65..−1.65, both because of which piece kinds they carry, not error). The exception is
`rebot`/`8x8`, which reports `dz_mm -1.66..+28.81` — one piece ends up resting *on the arm's base plate*
rather than on the board. The Newton port records an anomaly in the same single cell (`-1.72..-0.55`,
"which looks like a piece resting on the arm's base plate"); here it is larger. Neither port fixes it.

`reachable=` and `targets=` are also worth reading: they come from the lab's own reach and table-edge
filters, and they match the Newton port's exactly (`franka` on `8x8` gets `reachable=30/32 targets=36` in
both).

### How this port was verified against the Newton one

Three checks, in increasing order of how much they cover:

1. **The grasp math is byte-identical.** `genesis-world/scripts/crosscheck_grasps.py` dumps the whole
   planner fingerprint from each port and the two files diff clean — see [Package layout](#package-layout).
2. **The pieces settle to the same millimetre.** `dz_mm -1.65..-1.24` here against Newton's
   `-1.67..-1.25`, on the same scene. That covers the transcoded collision hulls, the computed inertia
   tensor, the authored mass, the board's substitute box, and the spawn clearance all at once — get any of
   them wrong and the number moves.
3. **The same seed draws the same episodes.** Running `--robot franka --scenario 4x4 --world-count 8
   --num-episodes 16 --seed 0` gives, as episode 4, `world 3 king piece_white_king_0 -> (+0.255,-0.363)`.
   The Newton README documents the same run's episode 4 as `world 3 king piece_white_king_0 ->
   (+0.255,-1.013)` — the same square, minus world 3's −0.65 m offset (Newton offsets its worlds in space;
   Genesis' batched envs share one frame). Same piece, same destination, same world, same episode index.
   That is what makes the pick success rates below a comparison of two *engines* rather than of two
   different task streams.

## Robot options

The four supported arms are the four with a real parallel-jaw gripper, matching the Isaac Lab picking
task's `CHESS_ROBOTS` and the Newton port's `ROBOT_OPTIONS`:

| `--robot` | source | arm + gripper DOFs | jaw span | reach | board distance |
|---|---|---|---|---|---|
| `franka` | newton-assets `franka_emika_panda/urdf/fr3_franka_hand.urdf` | 7 + 2 | 80 mm | 0.68 m | 0.45 m |
| `piper` | repo `assets/piper/piper_camera.usd` | 6 + 2 | 70 mm | 0.42 m | 0.30 m |
| `rebot` | newton-assets `seeed_rebot_devarm/urdf/seeed_rebot_devarm.urdf` | 6 + 2 | 90 mm | 0.44 m | 0.30 m |
| `yam` | Menagerie `i2rt_yam/yam.xml` | 6 + 2 | 79 mm | 0.42 m | 0.30 m |

```bash
$PY genesis-world/scripts/visualize_chess.py --robot piper --scenario 4x4 --viewer null --num-frames 120
$PY genesis-world/scripts/visualize_chess.py --robot rebot --scenario 4x4 --viewer null --num-frames 120
$PY genesis-world/scripts/visualize_chess.py --robot yam   --scenario 4x4 --viewer null --num-frames 120
```

Every geometric number in that table — jaw span, reach, board distance, and the `tcp_offset` /
`approach_axis` / `closing_axis` / `home_joint_pos` that do not appear in it — is carried over from the
Newton port verbatim, because they were measured on these exact asset files and do not change with the
physics engine. `load_robot()` re-checks the assumptions they rest on rather than trusting them: the DOF
count against the spec, and that every link a spec names survived the import.

`so101`, `ur10` and `flexiv_rizon` are **out of scope**. They still resolve through `robots.get_spec()`
(they are listed in `ROBOTS` behind `UNSUPPORTED_ROBOTS`) but they are not a `--robot` choice on either
script, nothing is verified against them, and the last two have no gripper at all.

## Chess layouts

The same five scenarios as `lab/` and `newton/`, selected with `--scenario`:

```bash
# one of every piece kind on a compact 3x3 board, sized so the shorter arms reach all of it
$PY genesis-world/scripts/visualize_chess.py --scenario pieces --viewer null --num-frames 120

# 1D: king, knight, rook vs rook, knight, king
$PY genesis-world/scripts/visualize_chess.py --scenario 1d --viewer null --num-frames 120

# pawn-only 3x3
$PY genesis-world/scripts/visualize_chess.py --scenario 3x3 --viewer null --num-frames 120

# 4x4 Mallett-Hill-Boyer minichess (default)
$PY genesis-world/scripts/visualize_chess.py --scenario 4x4 --viewer null --num-frames 120

# standard 8x8 opening, 32 pieces
$PY genesis-world/scripts/visualize_chess.py --scenario 8x8 --viewer null --num-frames 120
```

`--board-scale` stretches the squares; the default `auto` uses the per-scenario value from the lab config
— 1.4 everywhere except `8x8`, which stays at 1.0 because stretching a 0.48 m board by 1.4 pushes most of
it outside the Franka's reach. At 1.0 the 1.5x pieces fill 78–98 % of a 60 mm square, which is why every
other scenario is widened: 1.4 opens ~25 mm of room beside a piece for a 22 mm finger to descend into.

## Several worlds, several arms

`--world-count` replicates the scene. `--robot`, `--scenario` and `--board-scale` also take
comma-separated lists, which gives the worlds different content:

```bash
# four identical worlds, batched into four real Genesis envs
$PY genesis-world/scripts/visualize_chess.py --scenario 3x3 --world-count 4 --viewer null --num-frames 120

# a franka on 8x8 next to a yam on 1d
$PY genesis-world/scripts/visualize_chess.py --robot franka,yam --scenario 8x8,1d --world-count 2 \
    --viewer null --num-frames 120
```

Identical worlds become real Genesis envs. Worlds that differ **cannot**: `scene.build(n_envs=N)` batches
*the same scene* N times — an env is a copy, not a variant — so mixed runs are laid side by side inside a
*single* env at the offsets the batch would have used. `describe()` prints `genesis_envs=1` and says so
when that happens. This is the same fallback the Newton port makes for the same class of reason (there,
`SolverMuJoCo` refusing heterogeneous worlds).

One consequence worth knowing, because it is the opposite of Newton: **a batched Genesis env is not
offset in space.** `env_spacing` separates the envs for *rendering only*; `get_links_pos` returns
identical numbers for every env of an identical scene. So a world's `origin` is the zero vector in the
batched mode and the real side-by-side offset otherwise, and `ChessWorld` resolves the difference so
nothing downstream has to care.

## Rendering

With a display, `--viewer window` opens Genesis' interactive OpenGL viewer. Without one — this box has no
`$DISPLAY` — `--viewer gl` renders offscreen through the vendored pyrender rasteriser, which already
defaults to hardware EGL on Linux; there is no `PYOPENGL_PLATFORM` to set and no `--headless` flag,
because there is no on-screen path to opt out of. The startup line reports which it got:

```
  render: offscreen rasteriser (hardware EGL) res=(1280, 720)
```

`hardware EGL` versus `llvmpipe/software` comes from `scene.visualizer.is_software`, and is worth a look
if a render is unexpectedly slow.

```bash
# one PNG per frame plus an mp4
$PY genesis-world/scripts/visualize_chess.py --robot franka --scenario 4x4 --viewer gl \
    --save-images out/franka_4x4 --save-video out/franka_4x4.mp4 --num-frames 60

# a close-up of the 8x8 board
$PY genesis-world/scripts/visualize_chess.py --robot franka --scenario 8x8 --viewer gl --zoom 2.6 \
    --save-images out/franka_8x8 --num-frames 30
```

Those produce `frame_00000.png` upward and a h.264 mp4. Unlike the Newton port, **the mp4 works in this
interpreter** — it goes through `imageio-ffmpeg`, and without it the run degrades to PNGs with a warning
naming what is missing rather than producing nothing.

`--zoom`, `--camera=x,y,z` and `--camera-target=x,y,z` move the camera (the `=` form is required for a
negative coordinate, or argparse reads the leading `-` as an option). `--width`/`--height`/`--fps` size the
framebuffer and the timeline.

`--viewer gl` and `--viewer window` are about *where the frames go*, not about whether they are captured:
`--save-images`/`--save-video` build an offscreen camera either way, so `--viewer window --save-video
out.mp4` on a display both shows the run and records it. The combinations that cannot work are refused
rather than accepted and quietly ignored — `--viewer null` with a save flag, and `--viewer gl` with none
(it renders offscreen, so without a save flag there is nowhere for the frames to go).

**There is no scene export.** Genesis 1.3.3 has no USD, glTF or JSON writer of any kind, so the Newton
port's `--viewer usd|file|rerun|viser` have no counterpart here. Asking for one of those names says so and
exits rather than quietly doing something else. `--viewer null` renders nothing and is what the
measurements below use.

## Chess picking

Each episode hands every world one move — pick piece *i* up, put it on destination *j* — and
`robochess_genesis.pick.ChessPickTask` executes it with the GraspGen grasp the board actually leaves room
for. The nine-phase schedule (`pre_grasp, descend, close, lift, transfer, place, release, retreat,
settle`), the tolerances and the success criterion are ported from the Isaac Lab policy through the Newton
one: the piece must end within 20 mm of its destination square, within 10 mm of the surface, upright to
25 degrees, at rest, and out of the fingers.

```bash
$PY genesis-world/scripts/run_chess_pick.py --robot franka --scenario 4x4 \
    --world-count 8 --num-episodes 16 --seed 0 --viewer null
```

```
[episode   4] world 3 king   piece_white_king_0 -> (+0.255,-0.363)  success         phase=place     steps=266 place_err=   7.3mm grasp=0.941/pen=0.0mm
...
[INFO] 15 successes / 16 attempts (94%) in 64.3s (665 control ticks, 1330 frames)
[INFO] failures by cause: missed_target=1
[INFO] by piece kind: king=2/3, knight=4/4, pawn=9/9
[INFO] IK finished >1mm from the command on at least one world in 88 of 665 control ticks (normal when a leg's target is briefly outside the workspace)
```

An episode ends as `success`, or with the reason it did not. The outcome is latched as soon as the
schedule finishes, so a failure names the clause of the success predicate that rejected it —
`missed_target`, `off_surface`, `piece_tipped`, `not_released`, `still_moving` — or the event that ended it
early: `board_disturbed` (a piece that was not the target got knocked over), `piece_off_board` (the piece
left the table), `timed_out` (the step budget ran out mid-schedule). The line also reports the phase it
ended in, the placement error and the grasp quality. All worlds reset together, so a batch of N worlds runs
N episodes at a time.

`--balance-kinds` steers each episode toward the least-attempted piece kind, so a short run still covers
pawn, rook, knight, bishop, queen and king. `--debug` traces world 0 through every control tick. `--seed`
picks the move sequence.

Record one episode:

```bash
$PY genesis-world/scripts/run_chess_pick.py \
    --robot franka --scenario 4x4 --world-count 1 --num-episodes 1 --seed 0 --zoom 2.2 \
    --viewer gl --save-video out/franka_4x4_pick.mp4 --save-images out/franka_4x4_pick
```

At this seed that is 532 frames (a 708 kB mp4) of the arm lifting the black knight off the board and
setting it on the capture tray:

```
[episode   1] world 0 knight piece_black_knight_0 -> (+0.185,-0.433)  success  phase=release   steps=266 place_err=   5.5mm grasp=0.977/pen=1.5mm
```

**`--num-grasp-candidates 64` buys back most of the knight losses.** The grasp JSON holds 64 candidates
per piece; the default of 12 is there for parity with the Isaac Lab script. Raising it helps specifically
the knight — the one kind that is not a solid of revolution, so it gets no yaw spin with which to thread
the fingers between its neighbours:

```bash
$PY genesis-world/scripts/run_chess_pick.py --robot franka --scenario 8x8 \
    --world-count 6 --num-episodes 12 --seed 0 --num-grasp-candidates 64 --viewer null
```

`1d` 14/16 -> **16/16** with knight 1/3 -> 3/3; `8x8` 10/12 -> **11/12** with knight 1/2 -> 2/2. The
Newton port reports the same two improvements from the same flag, for the same reason.

`--balance-kinds` on `franka --scenario pieces` returns 15/16 with the draws spread as bishop 2/2,
king 1/2, knight 1/1, pawn 5/5, queen 4/4, rook 2/2 — every kind attempted at least once.

### Support matrix

Every cell below was run as

```bash
$PY genesis-world/scripts/run_chess_pick.py --robot <arm> --scenario <scenario> \
    --world-count 8 --num-episodes 16 --seed 0 --viewer null --no-visual
```

with `--world-count 6 --num-episodes 12` on `8x8`, everything else at its CLI default. Visualization
works in all twenty cells; the number is the **pick success rate**, and the Newton port's number for the
same command is given beside it so the two engines can be compared on what is provably the same episode
sequence.

| arm | `pieces` | `1d` | `3x3` | `4x4` | `8x8` |
|---|---|---|---|---|---|
| `franka` | **16/16** *(16/16)* | 14/16 *(16/16)* | **16/16** *(16/16)* | 15/16 *(16/16)* | **10/12** *(9/12)* |
| `piper` | 12/16 *(14/16)* | 12/16 *(14/16)* | **16/16** *(16/16)* | **16/16** *(15/16)* | 7/12 *(10/12)* |
| `rebot` | 5/16 *(6/16)* | 8/16 *(8/16)* | **16/16** *(15/16)* | 10/16 *(11-14/16)* | 1/12 *(1-3/12)* |
| `yam` | 3/16 *(12/16)* | 6/16 *(16/16)* | **16/16** *(16/16)* | 11/16 *(12/16)* | 3/12 *(9/12)* |

*(Newton port's figure in italics, from [`newton/README.md`](../newton/README.md).)*

### Notes on the matrix

* **Three of the four arms land within a couple of episodes of the reference**, and two cells beat it
  (`franka` `8x8`, `piper` `4x4`). `3x3` is 16/16 for every arm in both ports — it is the floor of both
  ports, pawns only, comfortably inside every arm's reach.
* **The `rebot` column is the one that had to be *fixed* rather than tuned.** It was 0/16 on every board
  until the collision-decomposition default was corrected (see [Notes for
  maintainers](#notes-for-maintainers)); it now tracks the Newton port cell for cell.
* **The `yam` column is this port's genuine shortfall** — `pieces` 3/16 against 12/16 and `1d` 6/16
  against 16/16. Its knights are 0/7 on `1d` and its tall pieces fail on `pieces`, while its pawn-only
  `3x3` is a clean 16/16, so what it loses is the narrow-neck and tall-piece grasps. See
  [Limitations](#limitations).
* **Read the weak cells as ±2.** `rebot`/`4x4` came back 14/16 during the gain sweep and 10/16 in the
  matrix run at the same setting; the Newton port reports the same instability on the same cell (it
  quotes 11-14/16 over ten runs). The strong cells are stable — `franka`/`4x4` reproduced 15/16 three
  times running, down to the per-kind split.
* **Per-kind counts are small-n.** The seed draws whatever it draws, so a cell's `rook=0/1` is one
  episode, not a verdict on rooks. `--balance-kinds` is the flag for probing a kind on purpose.
* **`franka --scenario pieces` is the strongest single result**: 16/16 with bishop 3/3, king 3/3, pawn
  5/5, queen 4/4, rook 1/1 — one of every kind, so the 7-9 mm necks of the tall pieces are all being
  pinched successfully. That is the same 16/16 and the same per-kind split the Newton port reports for
  the same command.

## Where this maps onto `lab/` and `newton/`

| Isaac Lab | Newton | Genesis |
|---|---|---|
| `zero_agent.py --task RoboChess-Visual-v0` | `newton/scripts/visualize_chess.py` | `genesis-world/scripts/visualize_chess.py` |
| `generate_chess_pick_demos.py` | `newton/scripts/run_chess_pick.py` | `genesis-world/scripts/run_chess_pick.py` |
| `chess/board.py` | reused through importlib | reused through importlib |
| `robot_configs.py` `ChessRobotSpec` | `robots.py` `NewtonRobotSpec` | `robots.py` `GenesisRobotSpec` |
| `FrankaChessSceneCfg` + `ChessPickEnvCfg` | `scene.py` `ChessScene` / `ChessWorld` | `scene.py` `ChessScene` / `ChessWorld` |
| `ChessPickPolicy` | `pick.py` `ChessPickTask` | `pick.py` `ChessPickTask` |
| differential-IK action term | `newton.ik` Levenberg-Marquardt | `RigidEntity.inverse_kinematics` (damped least squares) |
| PhysX articulation + rigid bodies | `SolverMuJoCo` + convex-hull colliders | Genesis rigid solver + convex-hull colliders |
| `ArticulationCfg` in a config tree | URDF/MJCF/USD into a `ModelBuilder` | URDF/MJCF/USD into a `gs.Scene` |
| baked piece USDs read directly | `newton.Mesh` shared across instances | **transcoded to OBJ + a one-link URDF** (`assets.py`) |
| `RecorderManager` -> HDF5 | not ported; outcomes are printed | not ported; outcomes are printed |
| USD stage / rerun / viser export | `--viewer usd\|rerun\|viser\|file` | **no equivalent** — Genesis has no scene export |

## Deliberate differences from the Isaac Lab task

These are the Newton port's list, and every one of them applies here for the same reason; the port
inherits both the decision and the measurement that justified it.

* **The board is a collider here.** Isaac Lab's chessboard is an `AssetBaseCfg` with no collision API and
  its pieces rest on the *table*; the shipped board USDs are zero-thickness quads, which no physics engine
  can rest a piece on, so this port adds a thin static box whose top face is exactly at the table top. The
  contact plane is identical; the friction the piece sees is 0.9 rather than the table's 1.0.
* **One friction number per surface.** Genesis, like Newton, takes a single `mu` where Isaac Lab has
  static and dynamic. Pieces use the static value (1.1), the table and board the dynamic one (0.9).
* **Rebot and YAM home postures were re-solved** for the URDF/MJCF sources, by the Newton port; the same
  files are loaded here, so the same postures apply.
* **Two departures in the pick schedule**, both because a solver-side IK tracks its command far more
  tightly than the differential-IK action term the Isaac Lab schedule was tuned against: a leg's last
  commanded pose is the goal rather than `(N-1)/N` of the way to it, and `transfer` ends at the carry
  height rather than a fixed 120 mm so the carry is level instead of a descending sweep across the pieces
  still standing.
* **The pieces get a real inertia tensor**, computed by summing the 16 CoACD hulls at unit density and
  rescaling to the authored mass. Overlapping hulls inflate a pawn's mass by 16 % if integrated directly
  (0.029466 kg against the authored 0.025362), and only the *ratio* survives the rescale, so the overlap
  cancels out. On the Genesis side this number is written into the generated URDF's `<inertial>` and
  `recompute_inertia=False` keeps it.

And three that are specific to this port:

* **The piece render meshes are decimated to ~6 000 triangles.** Newton builds one `newton.Mesh` per kind
  and instances it; Genesis builds the mesh per *entity*, so a 32-piece 8x8 board would be 32 separate
  copies of a 227 456-face mesh. The colliders are untouched — the physics is identical either way — and
  `--visual-faces 0` restores the full-resolution render mesh.
* **The Piper is loaded from the repo USD, not the Menagerie MJCF.** Both load cleanly in Genesis with the
  same link names and joint order; the USD is the asset the Isaac Lab visual task spawns and the one the
  Newton port loads, so all three ports show the same robot, and its fingers travel 50 mm each against the
  Menagerie model's 35 mm (the commanded 35 mm reproduces the real Piper's 70 mm span either way).
* **The "at rest" success clause reads the centre-of-mass velocity.** Genesis' `get_links_vel` defaults
  to the link *origin*; the Newton port reads `State.body_qd`'s first three entries, documented as the
  linear velocity of the body's centre of mass. The two differ by `omega x r`, which for a 50 mm piece
  only matters while it is tumbling — which is exactly what the clause exists to catch — so this port
  passes `ref="link_com"` to read the same point the reference does.
* **The gripper stiffness is per arm, and none of the four wants the Newton port's 600.** This is the one
  tuning number that did not transfer, and it was re-measured rather than guessed — 8 worlds x 16 episodes
  at seed 0, which is the configuration the matrix uses (a different `--world-count` draws a different
  episode set, so a value chosen at 4 worlds does not transfer either):

  | arm | `4x4` @ 300 / 600 / 1200 | `1d` @ 300 / 600 / 1200 | shipped |
  |---|---|---|---|
  | `franka` | **15** / 13 / 14 | **14** / 11 / 13 | 300 |
  | `piper` | 16 / 16 / 16 | 12 / 12 / 12 | 300 (indifferent) |
  | `rebot` | 8 / **14** / 7 | 9 / 9 / 8 | 600 |
  | `yam` | 2 / 7 / **10** | 5 / 5 / **6** | 1200 |

  A four-fold spread across four arms is not noise, and it has a mechanical reading: these are four
  different jaw mechanisms, and the joint stiffness that produces a given clamp force *at the pad* depends
  on the transmission between them. The YAM's pads move through a linkage and want the most; the Franka's
  direct-drive jaws want the least. Going too stiff is not merely wasteful — every piece here is a
  *tapered* shaft, so an over-squeezed grip extrudes the piece out of the jaws rather than crushing it
  (at 2000 the franka's rook drops to 3/7, visibly ejected). `GenesisRobotSpec.pick_gripper_kp` holds the
  per-arm value and `--gripper-ke` overrides it.
* **Imported actuator force ranges are cleared on every arm.** The four assets import with force limits
  sized against their own kp of 10-100 (the YAM at ±28/±10 N m, the Piper MJCF at ±100/±10), while this
  port drives at 600-4000 — so a 10 mrad tracking error already asks for more than the limit, and
  everything above the clamp is thrown away, including the gravity compensation. The Newton port clears
  the equivalent MuJoCo rows for exactly this reason.

## Package layout

```
genesis-world/robochess_genesis/
  board_layout.py   importlib shim onto lab's board.py + the table/board/tray constants
  gsmath.py         xyzw <-> wxyz bridge and the numpy pose algebra
  assets.py         USD -> OBJ/URDF transcode cache, board and table placement, contact budget
  robots.py         GenesisRobotSpec table, per-arm morphs, asset staging, GraspGen retargeting
  viewer_utils.py   viewer selection and CLI flags, offscreen PNG/MP4 capture
  scene.py          ChessScene / ChessWorld -- the assembled scene and its slot bookkeeping
  grasps.py         GraspGen JSON -> per-arm end-effector poses, re-scored against the board
  pick.py           the nine-phase IK-driven pick-and-place state machine
genesis-world/scripts/
  visualize_chess.py     capability 1
  run_chess_pick.py      capability 2
  crosscheck_grasps.py   proves this port's grasp math is the Newton port's, by diff
```

`crosscheck_grasps.py` is the one thing here that is a *test* rather than a capability, and
it exists because `grasps.py` is the half of each port with no engine in it -- GraspGen JSON
in, numpy out -- so it can be checked exactly rather than statistically. It dumps a
fingerprint of the planner (the per-arm retargeting matrix, the 24-point probe cloud, the
carry heights, and for every piece kind the chosen candidate, its score, its penetration and
all seven derived keypoints) against a fixed synthetic board:

```bash
NEWTON=/home/yizhou/Projects/newton/.venv/bin/python
GENESIS=/home/yizhou/Projects/genesis/.venv/bin/python
S=genesis-world/scripts/crosscheck_grasps.py

$NEWTON  $S --port newton  > /tmp/newton.json
$GENESIS $S --port genesis > /tmp/genesis.json
diff /tmp/newton.json /tmp/genesis.json && echo IDENTICAL
```

Two interpreters because the two packages cannot share a process (one needs `warp`, the
other `torch`+`genesis`, and they pin incompatible dependencies). **Last run: identical**,
all four arms, all six piece kinds, 23 683 bytes each. That is what makes the pick results
below a statement about the two *engines* rather than about two different planners.

Both scripts have detailed `--help`. The package is usable without them:

```python
import sys
sys.path.insert(0, "genesis-world")  # or the absolute path to <repo>/genesis-world

from robochess_genesis.scene import ChessScene

scene = ChessScene(robot="franka", scenario="4x4")
scene.finalize()          # builds the gs.Scene, applies the drive gains, homes the arm
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

`ChessWorld` is the bookkeeping a flat Genesis scene does not give you: which entity is which piece, which
env a world reads, where each square is in world space. `ChessPickTask` from `pick.py` drives it;
`run_chess_pick.py` is a thin wrapper around the two.

## Limitations

Found by the verification passes and **not** fixed. None of them crashes or hangs; they are honest
weaknesses of the port.

**Results you can misread**

* **Every cell of the matrix is a 12- or 16-sample estimate, and some cells move between identical runs.**
  Fixing the seed fixes the *move sequence* exactly (see [verification](#how-this-port-was-verified-against-the-newton-one)),
  but not the trajectory: Genesis' GPU reductions are not bitwise deterministic, and where a grasp is
  marginal that difference decides the episode. How much it moves is **scenario-dependent**, measured over
  three identical invocations at seed 0 each:

  | cell | three runs |
  |---|---|
  | `franka` `4x4` at the shipped gains | 15, 15, 15 — exactly repeatable, down to the per-kind split |
  | `franka` `4x4` at `--gripper-ke 600` | 13, 13, 13 — also exactly repeatable |
  | `franka` `1d` at the shipped gains | 13, 14, 14 |
  | `franka` `1d` at `--gripper-ke 600` | 12, 12, 13 |

  So read `4x4` as a number and `1d` as a number ±1. A cell that disagrees with the table by one episode
  has not necessarily regressed.
* **Piper, Rebot and YAM have exactly one legal destination on `4x4`.** `targets=1` in the `describe()`
  line: their 0.42-0.44 m reach plus the table-edge filter leaves a single square, so every episode for
  those three arms places on the same square. Franka gets 6. This is inherited from the lab's own target
  filter, not introduced here — the Newton port has the identical limitation.
* **`timed_out` is rare rather than informative.** Since episodes latch their outcome when the schedule
  completes, `timed_out` means the step budget ran out *mid*-schedule. A failing episode reports which
  clause of the success predicate rejected it: `missed_target`, `off_surface`, `piece_tipped`,
  `not_released`, `still_moving`, plus `board_disturbed` and `piece_off_board`.

**Physics**

* **The knight is the recurring weak kind, and it is a grasp-geometry limit rather than a physics one.**
  It is the one piece that is not a solid of revolution, so it gets no yaw spin with which to thread the
  fingers between its neighbours — the planner's 12 candidates simply may not contain one that fits.
  `--num-grasp-candidates 64` is the flag for it; the JSON holds 64.
* **The reBot and the YAM are both weak on `pieces` and `1d`, and the cause is now known.** It is not
  grasping — both arms close on the piece and lift it. It is the **carry**. `carry_height` caps the
  carried *piece's base* at `CARRY_REACH_FRACTION * reach`, but the hand grips a tall piece near its
  *top*: for a 140 mm king the hand ends up ~127 mm higher than the base, **278 mm above the board**. On a
  0.42 m arm reaching a far square that pose is outside the workspace, and what follows is worse than a
  missed pose — Genesis' damped-least-squares solve saturates a joint limit, the next tick warm-starts
  from the saturated posture, and the error compounds. Measured on `yam`/`pieces`, tracking across
  `transfer`: **14 → 50 → 114 mm**, with `joint6` pinned at its +2.094 rad limit from step 40 of the
  episode and `joint4` pinning during the carry.

  The Newton port commands the *same* unreachable pose and survives it: its Levenberg-Marquardt solve
  degrades gracefully, and on the same episode its error **shrinks** (29.6 → 16.6 → 5.9 → 4.7 mm) where
  this port's grows. That is the whole difference, and it is a property of the two IK solvers rather than
  of anything in this port's task logic.

  Seven interventions were measured against it and **none is shipped**, because each either failed or
  cost more elsewhere than it bought:

  | tried | result |
  |---|---|
  | finer CoACD decomposition (0.1, 0.05, even 347 geoms) | 4/16 either way — no |
  | IK iterations 24 → 50 | no change |
  | larger IK step / lower damping | same or worse |
  | soft joint limits (`respect_joint_limit=False`) | 3/16 — no |
  | holding the previous solution when IK is unreachable | 3 → 5/16 |
  | reachability-aware grasp selection | 3/16 — no |
  | capping the *hand* height instead of the piece base | +1–2 on the weak cells, but **−4 on `franka`/`1d`** and −5 on `piper` |

  The last one is the near miss: as a fraction of reach it still binds on the long arms, and reverting it
  was the right call. A per-arm cap would keep the `rebot` gain (+3 across four cells) but that is inside
  the ±2/cell noise band, so it is not worth the knob.
* **The reBot cannot place a king**, 0/6 on `1d` at every gripper stiffness tried — a specific instance of
  the above. The Newton port records the same weakness on the same asset and names `tcp_offset` as the
  suspect.
* **The most promising unexplored fix is the reBot's other asset.** The Newton port switched from the
  `seeed_rebot_devarm` URDF to the `usd_structured` layer beside it — same bodies, joints and masses, but
  an authored 8–12 part collision decomposition per link instead of a per-link convex hull — and went from
  ~6/16 to **16/16 on `pieces`, `1d`, `3x3` and `4x4`**. Genesis cannot currently load that asset, for two
  separate importer reasons, both diagnosed here:
    1. `genesis/utils/usd/usd_material.py:39` follows a shader input's connection and calls
       `UsdShade.Shader(...)` on the source. The mujoco-usd-converter connects each shader's
       `diffuseColor` to the **Material's own public interface input**, so the source is a `Material`, not
       a `Shader`, and it raises `RuntimeError: Accessed schema on invalid prim`. Flattening the stage,
       de-instancing the Material prims (they are `instanceable`, so the shaders are instance proxies and
       cannot be edited in place) and resolving the interface connections to literals clears this.
    2. After that, `_parse_scene` returns no links and `rigid_entity.py:752` raises `IndexError`. Not
       diagnosed further.

  Anyone picking this up should start there: it is worth ~10 episodes per cell on the `rebot`, far more
  than any tuning knob.
* **The YAM is this port's weak arm**, where the reBot is the Newton port's. It needs the stiffest gripper
  of the four (1200) and is still well short of the reference on the boards with tall pieces. Its MJCF
  puts the finger pads on linkage child bodies (`lf_down`/`rf_down`) that the Newton port collapses into
  the finger roots and Genesis keeps separate, so the two ports' notions of "the finger" differ even
  though the world-space geometry does not — a likely place to look, and one this port did not chase.
* **`8x8` is the hardest board for every arm**, for the reason you would expect: it is the one scenario
  that does not get the 1.4x board stretch, so it is 60 mm squares against an 80-90 mm open hand. The
  failures there are `board_disturbed` during `descend` and grasps with several millimetres of modelled
  neighbour penetration — the open fingers clipping a neighbour on the way in.

**Speed**

* **This port is slower than the Newton one per world**, and the gap closes as worlds are added because
  Genesis batches envs on the GPU. `visualize_chess.py --robot franka --viewer null --num-frames 120`,
  one world, same machine:

  | scenario | Genesis | Newton |
  |---|---|---|
  | `3x3` | 6.2 ms/frame | — |
  | `4x4` | 32.6 ms/frame | 13.4 ms/frame |
  | `8x8` | 132.4 ms/frame | 19.4 ms/frame |

  A frame is four 1/240 s substeps in both ports. The `8x8` gap is the honest worst case: 32 pieces at 16
  convex hulls each is 513 collision geoms, and Genesis' narrow phase feels that more than MuJoCo's does.
  Batching claws a lot of it back — four `3x3` worlds run at 18.7 ms/frame *in total*, i.e. 4.7 ms per
  world-frame against 6.2 for one — so prefer `--world-count 8` over eight separate runs.
* **Do not benchmark against a busy GPU.** Every number here was taken with the card otherwise idle;
  the same `franka` `4x4` cell measured 117.8 ms/frame while another process was resident and 32.6 ms
  once it left.
* **Build time, not step time, is the cliff for many colliders.** 16 hulls per piece is 513 collision
  geoms on an 8x8 board, and `_compute_collision_pair_idx` is `O(n_geoms^2)` numpy plus a Python loop.
  Measured cold (empty `~/.cache/quadrants` kernel cache) that is ~30 s; warm it is 3-5 s. It is not a
  hang.

**Rendering**

* **No scene export.** Genesis has no USD, glTF or JSON writer, so there is no equivalent of the Newton
  port's `--viewer usd` stage that you can open in usdview or Isaac Sim. Frames and video are the only
  artifacts.
* **The piece render meshes are decimated by default** (see [Deliberate
  differences](#deliberate-differences-from-the-isaac-lab-task)). `--visual-faces 0` restores them, at a
  real cost in build time and frame rate on the crowded boards.



## Notes for maintainers

Hard-won facts about this stack, each of which cost a debugging cycle. Several are the reason a line of
code looks the way it does.

* **Genesis re-origins a free body's link frame at its centre of mass, silently.** `align` defaults to
  "on" for a plain rigid entity, which moves the base link to the CoM *and* rotates it onto the inertia
  principal axes — for a pawn, +24.3 mm of z and a 24.8-degree spin. Nothing announces it, and
  `entity.get_pos()` still returns the un-aligned origin, so the discrepancy only shows up in the solver's
  own arrays: pieces resting on the board read back at z = 0.7947 instead of 0.7703. Every frame this port
  cares about — the GraspGen matrices, the square positions, the 10 mm "on the surface" success clause —
  is the piece's *authored* frame, so the piece morphs pass `align=False`.
* **`PXR_WORK_THREAD_LIMIT=1` is mandatory**, inherited from the Newton port: openusd's parser segfaults
  nondeterministically when it is allowed a thread pool, and both the piece transcoder and the Piper USD
  go through it. `board_layout.py` sets it at import and is the module every other one imports first.
* **`package://` does not resolve.** Genesis' URDF parser turns `package://franka_emika_panda/meshes/x.stl`
  into `<urdf dir>/franka_emika_panda/meshes/x.stl`, one directory too deep for both newton-assets URDFs.
  The parse fails, Genesis falls back to its legacy urdfpy parser, and *that* one raises
  `TypeError: Cannot cast array data from dtype('O')` out of `_init_dof_fields` — a stack trace with
  nothing in it about mesh paths. `robots.stage_urdf()` rewrites the references to absolute paths in a
  cached copy rather than touching the shared checkout.
* **`requires_jac_and_IK` defaults differently per importer, and getting it wrong fails one capability
  away.** `gs.morphs.URDF` and `gs.morphs.MJCF` default it to `True`; **`gs.morphs.USD` defaults it to
  `False`**. A USD arm left at the default loads, renders, and holds its home posture perfectly — and then
  the first pick tick raises `Inverse kinematics and jacobian are disabled for this entity`, several
  seconds into an episode and a whole capability away from the cause. That is precisely what happened to
  `--robot piper` here. `load_robot()` now re-checks the flag on the built entity, so the failure lands at
  load time with the reason attached.
* **`links_to_keep` and `merge_fixed_links` are URDF-only morph fields.** Passing either to
  `gs.morphs.MJCF` or `gs.morphs.USD` raises `Unrecognized attribute`. Neither needs them: MuJoCo welds
  jointless bodies rather than fixed-jointing them, and the USD importer keeps every rigid-body prim (with
  its full prim path as the link name, which is why `resolve_link` matches on the last path component).
* **An XML comment may not contain a double hyphen.** The generated piece URDFs are written by a Python
  f-string, and a comment mentioning `--visual-faces` made every one of them unparseable.
* **`scene.step()` updates the render state even with no camera and no viewer.** On the 4x4 board that is
  73.0 ms/frame against 22.9 with it off — a 3.2x tax for a picture nobody takes. `ChessScene.step()`
  passes `update_visualizer=False` unless something is going to look at the result, and only on the last
  substep of a frame when something is.
* **IK defaults to 50 random restarts, and a restart resamples every joint uniformly inside its limits.**
  That is right for "find me a posture reaching this pose" and wrong for a servo loop: a leg whose target
  drifts briefly out of reach comes back with an unrelated arm configuration. `pick.py` runs
  `max_samples=1`, which makes the call a plain warm-started descent. It costs nothing — the call is
  launch-bound at ~25 ms whatever the iteration budget (27.2 ms at 50x20 against 27.8 ms at 1x24).
* **IK failure is silent.** No exception, no warning; the only signal is `return_error=True`. Worse, a
  returned error of exactly 1e4 is a sentinel meaning "no sample ever improved on the seed", in which case
  the returned qpos is uninitialised memory. `pick.py` checks both and keeps the previous warm start on
  the sentinel.
* **`inverse_kinematics(dofs_idx_local=...)` is 0-based *entity*-local**, unlike almost every other
  `RigidEntity` method, which offsets by the entity's `dof_start`. Passing solver-global indices produces
  a bogus, exception-free result.
* **`enable_multi_contact=True` is load-bearing for grasping.** With it off a convex pair emits one
  contact point per finger, the grasp becomes a two-point pin, and the piece is dropped. It is on by
  default; do not turn it off even though the source labels it experimental.
* **Do not switch to `friction_cone=elliptic` naively.** With elliptic and the Newton constraint solver,
  Genesis auto-resolves `contact_resolution` to `signorini`, which loses a pinch grip. (The Newton port
  *does* run an elliptic cone with `impratio=50`; the two engines do not mean the same thing by it.)
* **`use_contact_island=False` is the single biggest speed win here** — 37.7 ms/frame to 20.1 on 4x4 over
  8 envs, with the settling unchanged. Contact islands earn their bookkeeping when contacts form separable
  clusters, and a chess board is one cluster: every piece touching one board box.
* **`max_collision_pairs` raises when it overflows** rather than silently dropping contacts, which is the
  opposite of the Newton port's contact budget and much easier to live with. `RigidOptions.max_contacts`
  is a *ceiling* that gets `min()`-ed with the auto-derived cap, so raising it does nothing; raise
  `max_collision_pairs` instead. Measured peak on 4x4: 210 broad-phase pairs and 166 contacts, against the
  4096-pair budget `assets.collision_pair_budget()` sizes.
* **Genesis convex-hulls robot collision meshes on import, and its default does too much of it.** The
  reBot URDF points its `<collision>` tags at the same full-resolution STLs as its `<visual>` tags
  (364 392 vertices over ten links); the Newton port has to approximate them by hand, and Genesis'
  `decompose_robot_error_threshold=inf` default — "convexify, never decompose" — already does it. But a
  reBot finger is a concave **wedge**, and its single convex hull *bridges the jaw slot*: the jaws bottom
  out on their own hulls 31 mm too early, pinch the piece's 47 mm flare instead of its 11 mm neck, and it
  slips out on the lift. Every episode ended `missed_target`; the arm was **0/16 on every board** while
  loading, homing and rendering perfectly. Giving the threshold a finite value splits each finger into 10
  hulls and takes the arm to 8/8, closing to a 0.233 jaw fraction against the Newton port's 0.25. The
  threshold is self-selecting — the franka's already-convex fingers come out with an unchanged geom count
  — which is why `GenesisRobotSpec.collision_decompose` sets it for every arm rather than just the one.
  Note this is the **opposite** correction from the Newton port's: there the meshes were too detailed,
  here the hull is too coarse.
* **`gs.init()` raises if called twice**, and `gs.destroy()` is the way back. `scene.init_genesis()` is
  idempotent so that importing the package into a live session does not fight whatever initialised it.
* **The baked piece assets are a wrapper plus a payload.** `generated/s150/pawn.usd` is 11 kB — it carries
  the 16 collision hulls inline and *references* `assets/chess/pawn.usdc`, the 10 MB render mesh. Anything
  that fingerprints the assets has to follow the reference or a re-bake of the payload leaves the wrapper's
  mtime untouched; `assets._stamp()` resolves them with `Sdf.Layer.FindOrOpen`, which reads the wrapper
  without composing the payload in.
* **`max_collision_pairs` is a per-environment budget**, not a per-scene one: Genesis clamps it to the
  env's own possible-pair count and shapes the contact cache `(n_possible_pairs, n_envs)`. Multiplying it
  by the batch size only inflates a number that is about to be clamped.
