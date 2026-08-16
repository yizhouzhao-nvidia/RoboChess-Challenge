# RoboChess Isaac Lab task

This package contains the first manager-based Isaac Lab migration for RoboChess.

## Install

Activate the existing Isaac Lab environment, then install this package:

```bash
cd /home/yizhou/Projects/RoboChess-Challenge/lab
uv pip install -e . --python /home/yizhou/Projects/IsaacLab/env_isaaclab/bin/python
```

## Visual inspection

Launch one stationary, single-arm scene:

```bash
cd /home/yizhou/Projects/RoboChess-Challenge/lab
/home/yizhou/Projects/IsaacLab/env_isaaclab/bin/python scripts/zero_agent.py \
  --task RoboChess-UR10-Visual-v0 \
  --num_envs 1
```

The task includes one standard Isaac Lab UR10, a table, and a 4x4 minichess board. It has no XR, teleoperation, or bimanual setup. The task uses the project chessboard and chess-piece USD/USDC assets under `assets/chess`. It remains a visual, single-arm setup without teleoperation.

## Robot options

Use `--robot` with any chess layout:

```bash
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --robot so101 --chess_scenario 4x4
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --robot piper --chess_scenario 4x4
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --robot ur10 --chess_scenario 4x4
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --robot flexiv_rizon --chess_scenario 4x4
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --robot rebot --chess_scenario 4x4
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --robot yam --chess_scenario 4x4
```

SO-101 and Piper use local USDs in `assets/`. Flexiv Rizon, Rebot, and YAM use the URLs from their existing configurations. For offline Rebot or YAM use, set `REBOT_USD_PATH` or `YAM_USD_PATH` to a local USD before launching.

## Chess layouts

Select a setup using `--chess_scenario`:

```bash
# 1D: king, knight, rook vs rook, knight, king
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --chess_scenario 1d

# Pawn-only 3x3
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --chess_scenario 3x3

# 4x4 Mallett-Hill-Boyer minichess (default)
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --chess_scenario 4x4

# Standard 8x8 opening
python scripts/zero_agent.py --task RoboChess-UR10-Visual-v0 --chess_scenario 8x8
```

## Franka chess picking

`RoboChess-Franka-Chess-IK-Abs-v0` is a manipulation task, not just a visual one: a
Franka Emika Panda picks the chess piece a command term selects and puts it on a
target square (or in the capture tray). Grasps come from
[GraspGen](https://github.com/NVlabs/GraspGen), and a scripted state machine turns
them into demonstrations recorded through Isaac Lab's `RecorderManager`.

See [docs/franka_chess_picking.md](docs/franka_chess_picking.md) for the full pipeline.
The short version:

```bash
PY=/home/yizhou/Projects/IsaacLab/env_isaaclab/bin/python

# 1. bake rigid-body chess pieces at 1.5x scale (+ meshes for GraspGen)
$PY lab/scripts/prepare_chess_assets.py --scale 1.5

# 2. ask GraspGen how a Franka should grasp each piece (GraspGen's own venv)
/home/yizhou/Projects/GraspGen/.venv/bin/python lab/scripts/graspgen_chess_grasps.py

# 3. generate and record demonstrations
$PY lab/scripts/generate_chess_pick_demos.py --headless --num_envs 16 --num_demos 100 \
    --chess_scenario 4x4 --dataset_file ./lab/datasets/franka_chess_pick_4x4.hdf5

# 4. replay the dataset and render it
$PY lab/scripts/render_chess_demo.py --headless --demos 0 1 2 --video \
    --dataset_file lab/datasets/franka_chess_pick_4x4.hdf5
```

Add `--chess_scenario 8x8 --zoom 1.45` for full-board chess moves (and for bishops
and queens, which the 4x4 minichess setup has none of).

### Other arms

The task is robot-agnostic. `--robot` selects any of `franka`, `piper`, `rebot` or
`yam`; all four load straight from their upstream URLs, nothing is vendored:

```bash
$PY lab/scripts/probe_robot.py --headless --robot rebot      # joints, prims, gripper axes
$PY lab/scripts/probe_reach.py --headless --robot rebot      # can it reach the board?

$PY lab/scripts/generate_chess_pick_demos.py --headless --robot rebot \
    --chess_scenario pieces --balance_kinds --num_demos 10 \
    --dataset_file ./lab/datasets/chess_pick_rebot.hdf5
```

`--chess_scenario pieces` is a compact 3x3 board holding one of every piece kind,
sized so the shorter arms can reach all of it; `--balance_kinds` steers each episode
toward the kind with the fewest recorded demos so a short run still covers pawn,
rook, knight, bishop, queen and king. See
[docs/how_this_was_built.md](docs/how_this_was_built.md) for what each arm needed.
