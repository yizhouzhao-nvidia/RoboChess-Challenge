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
