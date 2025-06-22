from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from WheelchairRL.assets.robots.carter_cfg import CARTER_CFG


@configclass
class ArucoTaskEnvCfg(DirectRLEnvCfg):
    # Environment
    decimation = 4
    episode_length_s = 10.0
    action_space = 2
    observation_space = 4
    state_space = 0

    # Simulation configuration
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot configuration
    robot_cfg: ArticulationCfg = CARTER_CFG.replace(prim_path="/World/envs/env_.*/Robot")


    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Custom parameters
    action_scale = 2.0
