import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.config import Config
from src.environments.simple_humanoid import SimpleHumanoidEnv
from src.environments.mujoco_humanoid import MuJoCoHumanoidEnv


def create_env(config: Config):
    if config.env_name == "simple_humanoid":
        env = SimpleHumanoidEnv()
    elif config.env_name == "mujoco_humanoid":
        # env = MuJoCoHumanoidEnv()  # this didn't work because it doesn't behave sufficiently like any other gym environments
        env = gym.make("Humanoid-v4", render_mode="rgb_array")
    else:
        raise ValueError(f"Unknown environment: {config.env_name}")
    
    if config.monitor:
        env = Monitor(
            env,
            filename=str(config.experiment_path),
        )
    
    if config.record_video:
        env = RecordVideo(
            env,
            video_folder=str(config.videos_path),
            episode_trigger=lambda step_id: step_id % 100 == 0,
            step_trigger=None,
            video_length=0,
            name_prefix="rl-video",
            disable_logger=False,
        )

    env = make_vec_env(
        lambda: env,
        n_envs = 1,
        seed = config.seed,
    )
    return env
