# RoboChess Challenge

> [!WARNING]
> UNDER DEVELOPMENT: This project is actively being built and may change frequently.


![ChessRobot](images/title.png)

Robotic chess manipulation in [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim), built around the [LeRobot SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm. The project combines a cuMotion-based motion-planning extension, an Isaac Lab task scaffold for RL, and a set of chess scenarios (from a 1D toy board up to full 8x8 chess) used to develop and test pick-and-place manipulation.

## Features

- **cuMotion Plan extension** ([exts/cumotion.plan](exts/cumotion.plan)) — an Isaac Sim Kit extension that drives the SO-101 arm with collision-free IK and motion planning (NVIDIA [cuMotion](https://github.com/nvidia-isaac/cumotion)) to pick up and place chess pieces.
- **Chess scenarios** ([so101.py](exts/cumotion.plan/CuMotion_Plan_python/cumotion/so101.py)) — selectable board setups for incrementally harder manipulation tasks:
  - `1D chess` — a single rank of 6 pieces, for basic pick-and-place testing.
  - `3x3 chess` — a 3x3 board with pawns only.
  - `4x4 chess` — the Mallett–Hill–Boyer [minichess](https://en.wikipedia.org/wiki/Minichess) variant (king, 2 knights, rook, 4 pawns per side).
  - `8x8 chess` — standard full chess setup.
- **Newton example** ([newton/example_so101.py](newton/example_so101.py)) — a standalone SO-101 example using the [Newton](https://github.com/newton-physics/newton) physics engine.
- **Isaac Lab task** ([lab/ChessRobot](lab/ChessRobot)) — an Isaac Lab extension (`Template-Chessrobot-Direct-v0`) scaffolding the SO-101 chess environment for reinforcement learning.
- **Franka chess picking** ([lab/docs/franka_chess_picking.md](lab/docs/franka_chess_picking.md)) — a Franka Emika Panda that picks and places chess pieces in Isaac Lab, grasps chosen by [GraspGen](https://github.com/NVlabs/GraspGen), with a scripted state machine generating demonstrations recorded through Isaac Lab's `RecorderManager`.

## Scenarios

| 3x3 chess | 4x4 chess | 8x8 chess |
|---|---|---|
| ![3x3 chess](images/3x3%20chess.png) | ![4x4 chess](images/4x4%20chess.png) | ![chess](images/chess.png) |

## Repository layout

```
exts/cumotion.plan/   Isaac Sim Kit extension (UI + cuMotion-driven SO-101 pick-and-place)
assets/                Robot, chess piece, and board USD assets
lab/source/robochess/  Isaac Lab manager-based tasks (visual scenes + Franka picking)
lab/scripts/           Asset baking, GraspGen inference, demo generation, replay/render
lab/docs/              Pipeline documentation
newton/                Standalone SO-101 example using the Newton physics engine
scripts/               Helper scripts (e.g. cuMotion)
package/               Robot meshes (e.g. .dae)
```

## Installation

Full setup notes (Isaac Sim, cuMotion, Isaac Lab) live in [setup.md](setup.md). In short:

1. Install [Isaac Sim](https://developer.nvidia.com/isaac/sim) and point `ISAAC_SIM_PATH` at it.
2. Install [cuMotion](https://github.com/nvidia-isaac/cumotion) into the Isaac Sim Python environment.
3. (Optional, for RL) install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and the `ChessRobot` task package from `lab/ChessRobot`.

See [setup.md](setup.md) for the exact commands (Linux and Windows/PowerShell).

## Usage

Start Isaac Sim with the `cumotion.plan` extension enabled:

```sh
& "$env:ISAAC_SIM_PATH\isaac-sim.bat" --ext-folder exts --enable cumotion.plan
```

From the extension UI, load one of the chess scenarios (default is `4x4 chess`, configurable in [load_example_assets](exts/cumotion.plan/CuMotion_Plan_python/cumotion/so101.py)) and use the panel to plan and execute pick/place motions with the SO-101 arm.

For the Isaac Lab RL task:

```sh
python lab/ChessRobot/scripts/zero_agent.py --task=Template-Chessrobot-Direct-v0
```

## AI coding agents

Agent configuration for this repo lives in the [.agent](.agent) folder. `.claude` and
`.codex` are symlinks to it, so Claude Code and Codex share one source of truth.

```
.agent/
├── settings.json   Permissions and tool settings
├── agents/         Subagent definitions (e.g. isaaclab)
├── skills/         Skills
└── commands/       Slash commands
```

- **`isaaclab`** ([.agent/agents/isaaclab.md](.agent/agents/isaaclab.md)) — Isaac Lab
  specialist for task configs, MDP terms, scene/asset setup, and the scripts under `lab/`.

If you are Claude Code or Codex working in this repo, look in `.agent/` for the available
agents, skills, and commands before starting a task.

## Status

Active challenge project.
