# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

try:
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    @configclass
    class ChessrobotPhysicsCfg(PresetCfg):
        default: PhysxCfg = PhysxCfg()
        physx: PhysxCfg = PhysxCfg()
        newton_mjwarp: NewtonCfg = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=5,
                nconmax=3,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
        )

    _PHYSICS_CFG = ChessrobotPhysicsCfg()
except ImportError:
    _PHYSICS_CFG = None



@configclass
class ChessSceneCfg(InteractiveSceneCfg):
    num_envs=16
    env_spacing=4.0
    replicate_physics=True



@configclass
class ChessrobotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=_PHYSICS_CFG)

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = ChessSceneCfg() #InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]