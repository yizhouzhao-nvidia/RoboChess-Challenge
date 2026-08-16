# How the chess-picking pipeline was built

A step-by-step account of building [Franka chess picking](franka_chess_picking.md),
written so it can be re-run, audited, or extended to another robot. The emphasis is
on the *method* — especially how each problem was diagnosed — because most of the
elapsed time went into four bugs, none of which were where they first appeared.

Environments used throughout:

```bash
LAB=/home/yizhou/Projects/IsaacLab/env_isaaclab/bin/python   # Isaac Lab 3.0 / Isaac Sim 6.0
GG=/home/yizhou/Projects/GraspGen/.venv/bin/python           # GraspGen (CUDA, PyTorch 2.8)
```

---

## Step 0 — Establish the ground truth before writing anything

Before any code, three facts were measured rather than assumed. Each of them would
have silently corrupted everything downstream.

**Piece dimensions.** Load the shipped `assets/chess/*.usdc` and print bounding
boxes. A pawn is 47 mm tall and 31 mm across the base; the Franka gripper opens to
80 mm with 54 mm fingers. That single measurement decided the piece scale.

**Quaternion convention.** Isaac Lab 3.0 uses **(x, y, z, w)** — 2.x used (w, x, y, z).
Check `isaaclab/utils/math.py` docstrings and `AssetBaseCfg.InitialStateCfg.rot`
(default `(0,0,0,1)`), don't guess.

**Gripper closing axis.** Spawn the robot, step the sim, and print each finger's
pose *in the hand frame* with the gripper open and closed:

```python
pos_b, quat_b = math_utils.subtract_frame_transforms(
    hand_pos_w, hand_quat_w, finger_pos_w, finger_quat_w)
```

The Franka gives `(0, ±0.04, 0.0584)` — it closes along the hand's **Y**. GraspGen
closes along **X**. That 90° is the difference between a working grasp and a gripper
that closes on air.

> **Method note.** All three were established with throwaway probe scripts in a
> scratch directory, not in the repo. A probe that launches Isaac Sim, prints ten
> numbers, and exits is worth far more than a careful reading of the docs.

---

## Step 1 — Make the pieces simulatable

`lab/scripts/prepare_chess_assets.py` — needs only `pxr`, `trimesh` and `coacd`, so
it runs in ~2 minutes without launching Isaac Sim.

```bash
$LAB lab/scripts/prepare_chess_assets.py --scale 1.5
```

The shipped pieces are render-only 200k-triangle meshes. For each one the script:

1. Extracts and concatenates all `UsdGeom.Mesh` prims, baking their local transforms.
2. Normalises the frame — XY centred on the piece axis, `z = 0` at the base plane.
   *Both the simulator and GraspGen must agree on this frame*, so the same
   normalisation is exported as `.obj` and baked into the USD.
3. Applies the scale (default 1.5x).
4. Runs **CoACD** convex decomposition into 16 hulls. A single convex hull fills in
   the neck of the piece, which is exactly the feature the gripper grips.
5. Writes a USD: a rigid-body root with `RigidBodyAPI` + `MassAPI` (mass from mesh
   volume x density), the hulls as `CollisionAPI` + `MeshCollisionAPI(convexHull)`
   children, and the *original* mesh referenced for visuals so nothing is duplicated.

**Verification, not faith.** Reopen each generated USD and compare the visual and
collision bounding boxes. They agreed to within 0.7 mm. Then drop one piece alone
onto a table in an empty scene and confirm it settles upright at the expected height.

---

## Step 2 — Ask GraspGen how to grasp them

```bash
$GG lab/scripts/graspgen_chess_grasps.py
```

Runs the `franka_panda` model (diffusion generator + discriminator), 2000 samples per
piece, ~0.3 s each. Then — and this is the part that matters — **filter and re-rank**,
because raw GraspGen output is not directly usable on a board:

| filter | why |
|---|---|
| approach within 75° of straight down | the table blocks everything else |
| no gripper point below the board plane | it would collide with the board |
| piece fits inside the finger stroke | measured in the volume the fingers sweep |
| discriminator score > 0.5 | drop low-confidence samples |

GraspGen's argmax is almost always a **near-horizontal side grasp** of the shaft:
excellent for a floating object, terrible on a populated board. So the ranking
penalises tilt:

```
rank_score = score - tilt_weight * max(0, tilt_deg - preferred_tilt_deg) / 90
```

Chess pieces (except the knight) are solids of revolution, so each surviving grasp is
rotated into a canonical azimuth and the free yaw is deferred to the motion generator.

**Verification.** The script emits a per-piece PNG and a summary figure. Looking at
them immediately showed the strategy was coherent — down from above, pinching the
shaft under the head — and would have immediately shown if it were not.

---

## Step 3 — Build the environment

`lab/source/robochess/tasks/manager_based/chess/`. Design decisions worth repeating:

* **Put the task in a command term, not the script.** `ChessMoveCommand` samples
  "move piece *i* to destination *s*" per episode. The goal then travels with the
  environment — into the observations, into the `success` termination, and into any
  future policy — instead of living in a script that only the data generator runs.
* **Define each predicate once.** `piece_grasped` is an observation term; the
  `success` termination *and* the demo state machine both call it. When it was wrong
  (twice, see below) it was wrong in exactly one place.
* **Make sloppiness a failure.** `board_disturbed` fails the episode if any other
  piece is knocked over. A demo that completes the move but barges a neighbour is
  worse than useless for imitation.
* **Separate geometry from layout.** `board.py` maps all four scenarios onto one
  (file, rank) grid with a `board_scale`, so the board can be stretched without
  touching the pieces.

### Asset-level surprises

Two Isaac Sim 6.0 issues needed local workarounds, both in `robot_configs.py`:

1. `isaaclab_assets.robots.franka` still points at the 5.0 `panda_instanceable.usd`,
   which 404s. The 6.0 asset is `franka_panda.usda`; joint and body *names* are
   identical, only the path changed.
2. The 6.0 asset nests every link inside its parent under a `Geometry` scope. So
   (a) `FrameTransformerCfg` needs full prim chains, and (b) `disable_gravity` only
   reaches `panda_link0`, because `modify_rigid_body_properties` stops descending at
   the first rigid body it finds. **Diagnosed by counting**: 11 rigid bodies, 1 with
   the flag set. Fixed with stiffer position gains, which is closer to the real
   robot's gravity-compensated controller anyway.

---

## Step 4 — Generate and record trajectories

```bash
$LAB lab/scripts/generate_chess_pick_demos.py --headless --num_envs 16 --num_demos 100
```

A vectorised state machine over N environments:

```
pre_grasp -> descend -> close -> lift -> transfer -> place -> release -> retreat -> settle
```

Three things make it work:

* **Each leg waits for the arm to arrive**, it does not just run out its clock.
  Differential IK takes one Jacobian step per tick; a purely time-triggered schedule
  closes the fingers wherever the arm happens to be.
* **The place pose is measured, not assumed.** The hand-to-piece transform is
  captured after the fingers close and refreshed after the lift and the carry, so
  slip is compensated before the piece is set down.
* **Grasps are re-scored against the live board.** Every stored candidate (and, for
  solids of revolution, 16 azimuths) is checked by transforming the open fingers into
  the world and measuring how far they dip into neighbouring pieces.

Recording is Isaac Lab's stock `RecorderManager`: set
`env_cfg.recorders = ActionStateRecorderManagerCfg()` with
`DatasetExportMode.EXPORT_SUCCEEDED_ONLY` and the env does the rest on auto-reset —
it reads the `success` termination itself.

---

## Step 5 — Replay to prove the dataset

```bash
$LAB lab/scripts/render_chess_demo.py --headless --demos 0 1 2 --video
```

Restores each episode's initial state and re-executes its recorded **actions**. This
is a real test, not a screenshot generator: if the dataset were missing something,
the replay would diverge. It reports the drift per demo (typically 0.2–2 mm).

---

## The debugging method, and the four bugs

Every one of these presented somewhere other than its cause. The pattern that found
all of them: **instrument the state machine to print one line per phase transition**
(tracking error, gripper opening, piece height, distance to target, current goal),
then run one environment and read the trace.

| symptom | actual cause |
|---|---|
| Every piece topples on reset; board explodes | A yaw quaternion written `(cos, 0, 0, sin)` — wxyz — spawned everything 180° about X. Isaac Lab 3.0 is **xyzw**. |
| Fingers close 20–56 mm short of the grasp | Time-triggered phases + 30–50 mm steady-state IK error from `disable_gravity` not propagating. |
| Piece "placed" hundreds of mm off target | The place pose was never derived, because the *hold* predicate was false. Twice: (a) per-finger opening test — the two fingers close asymmetrically on an off-centre grasp; (b) TCP-vs-piece-*axis* test — the knight is gripped off-axis by the head. |
| Carried piece drags a king off the board | The carried piece hangs below the hand, so the carry height must clear the *tallest piece*, not be a fixed constant. |

Two general lessons:

**A false negative in a gate looks like a downstream failure.** The hold predicate
gates when the place pose gets computed. When it wrongly said "not holding", the arm
carried the piece to the lift pose and dropped it there — which reads as a *placement
accuracy* problem, so debugging starts several stages too late. If a stage's output
looks wrong, check whether the stage even ran.

**Turning on a stricter check can lower your metric, and that is good.** Adding
`board_disturbed` dropped the measured success rate from 95% to 63%. Nothing got
worse; demos that knocked pieces over had been counted as successes all along. Report
numbers with the check that was active, and never compare across different checks.

**Measure per seed.** Success varies 75–92% by seed on the same code, because which
(piece, destination) pairs get drawn dominates. Comparing two configurations measured
at different seeds is meaningless — a mistake worth avoiding by fixing the seed set
up front.

### The auto-reset trap

`ManagerBasedRLEnv.step()` **resets terminated environments inside the step**. So by
the time the caller sees the `done` flags:

* the command term has already resampled — the commanded piece is the *next*
  episode's,
* the termination manager has already been recomputed — `get_term("success")`
  describes the fresh episode,
* every asset pose is the post-reset one.

Anything a training or generation loop reads after `step()` about the episode that
just ended is therefore wrong, silently and plausibly. This produced three rounds of
incorrect per-piece statistics here, each of which *looked* right: totals matched,
only the distribution was wrong. It was caught by reading the HDF5, not by inspecting
the counter.

Two rules that fall out:

1. **Snapshot per-episode facts when the episode is planned**, never at its end.
2. **Do not attribute an aggregate to a batch.** The recorder's exported-success
   count is authoritative in total but cannot be split across environments that
   terminated in the same step. After four attempts at making that attribution work,
   the honest fix was to delete it: coverage is now read from the dataset
   (`lab/scripts/dataset_summary.py`) and the kind-balancing steers on *attempts*,
   which come from the plan-time snapshot and are exact.

A corollary worth internalising: a metric computed inside the loop that agrees with
the dataset on totals but not on breakdown is a strong signal you are reading
post-reset state.

### When the steering itself is the bug

Balancing coverage by "pick the kind with the fewest successes" fixates. A kind the
arm cannot pick stays scarcest forever and absorbs every episode: one run spent 182
of 260 attempts on a single piece and returned a 2% success rate — worse than the
buggy version it replaced, which had rotated by accident. Round-robin on attempts
cannot fixate and needs no success signal at all.

---

## Extending to another robot

The environment is robot-agnostic through `ChessRobotSpec` in `robot_configs.py`.
Four arms are configured, all loaded straight from their upstream URLs — nothing is
vendored into the repo:

| | franka | piper | rebot | yam |
|---|---|---|---|---|
| IK body | `panda_hand` | `link6` | `gripper_end` | `link_6` |
| prim layout | nested | flat | nested | under `arm/` |
| approach axis | +Z | +Z | **+X** | +Z |
| closing axis | Y | X | Y | X |
| TCP offset (m) | 0.1034 Z | 0.125 Z | −0.015 X | 0.115 Z |
| stroke | 80 mm | 70 mm | 90 mm | 90 mm |
| board distance | 0.45 m | 0.30 m | 0.30 m | 0.30 m |

The procedure, and it is worth following in order:

1. **Probe the asset** (`lab/scripts/probe_robot.py`): joint names and limits, body
   names, the prim path of the end-effector, and — with the gripper driven to both
   ends of its stroke — the finger offsets in the EE frame. That yields the closing
   axis, the approach axis, the TCP offset and the usable stroke.
   *Read these from the physics view, never from USD xforms*: USD holds the
   **authored** pose and goes stale the moment the simulation steps.
2. **Check the stroke against the grasps.** The chess grasps span 19–43 mm.
3. **Fill in a `ChessRobotSpec`**, then **check the reach** with
   `lab/scripts/probe_reach.py`, which drives the task's own IK to a grid of top-down
   poses over the board and reports the steady-state TCP error at each. That sets
   `reach`, `board_distance` and `home_joint_pos` — and catches gain problems before
   they look like grasp problems.
4. **Verification ladder**: scene settles → IK tracks to a few mm → one episode traces
   cleanly → batch success rate → replay drift.

The GraspGen grasps themselves are *not* regenerated per robot. GraspGen ships models
for three grippers only, and the chess grasps are top-down pinches of a shaft — a
property of the piece and of parallel-jaw geometry. They are retargeted by aligning
each robot's TCP and closing axis onto the GraspGen frame.

### Three more bugs, and why each hid

* **The retarget rotation was transposed.** Building it column-wise instead of
  row-wise gives a matrix that is *correct up to ±90° about the gripper's own
  symmetry axis* for any arm whose approach is +Z — so the Franka, Piper and YAM all
  worked, and only the reBot (approach +X) exposed it, by pointing its gripper
  sideways. Caught in seconds by a numeric check — retarget each stored grasp, then
  assert the gripper's approach direction and TCP still match GraspGen's — rather
  than by watching the arm.
* **YAM's gripper would not open.** Its two finger meshes overlap in the rest pose,
  so with self-collisions enabled PhysX fights the drive and the joints move ~1 mm
  however hard they are pushed. Disabling self-collisions fixed it. Also, its joint
  convention is the reverse of what the upstream comment implies: 0 is *wide open*.
* **Piper and YAM sagged a constant ~50 mm at every target.** Distance-independent
  error means gravity against an *effort* ceiling, not reach — their shipped
  `effort_limit_sim` (50 and 20 N·m) binds long before the position gain does.

A related trap in the probe itself: the IK drives the **end-effector body**, but a
target is where the **TCP** should land. Commanding the body straight to the target
asks the gripper to go through the table, and reads as an unreachable pose. Back the
command off by the tool offset.

### The last centimetre has to be measured by trying

`probe_robot.py` locates the finger *bodies* from physics, which fixes the approach
axis, the closing axis and the stroke exactly. It cannot locate the **pad face** the
gripper actually grips with — that is somewhere along a finger whose geometry is only
available in the (stale) authored pose.

YAM made the cost of guessing concrete: with `tcp_offset` set from finger geometry it
scored **0 successes in 140 attempts**, the gripper ramming each piece on descent and
then closing on air.

Bracketing at 16 attempts per value appeared to settle it — 0.09 m and 0.145 m gave
0 successes, 0.13 m gave 2 — but that conclusion did not survive. Pooling every run
at 0.13 m gives **4 successes in 416 attempts, ~1%**; under that rate, seeing 2 in 16
has probability 0.01, and the promising bracket was simply a lucky sample.

Two lessons, and the second is the one that cost real time:

* **Bracket rather than reason** when a quantity is cheap to test. The argument that
  preceded the bracket was wrong about the *direction*: the symptom (gripper too
  deep) suggested a shorter offset, and the answer was longer.
* **Size the bracket to the effect.** 16 attempts cannot distinguish 0% from 12%.
  Comparing candidate values at a per-run success rate in the low tens of percent
  needs on the order of 100 attempts each, not 16 — otherwise the bracket manufactures
  a winner out of noise, which is exactly what happened, and a wrong value then gets
  written into the config with a confident comment.

YAM remains unresolved at ~1%: its gripper actuates and its IK tracks to 6 mm, but the
grasp itself does not hold.
