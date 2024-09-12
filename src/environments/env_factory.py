import gymnasium as gym
from gymnasium import envs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder

from src.config import Config
from src.environments import simple_humanoid


def make_env(config: Config):
    if config.env_name == "simple_humanoid":
        env = simple_humanoid.SimpleHumanoidEnv()
    else:
        if config.env_name in envs.registry.keys():
            env = gym.make(config.env_name, render_mode="rgb_array")
        else:
            raise ValueError(f"Unknown environment: {config.env_name}")
    env = Monitor(env)
    return env


def create_env(config: Config):
    env = make_vec_env(
        env_id=lambda: make_env(config),
        n_envs=config.n_envs,
        seed=config.seed,
    )
    print(config)
    if config.record_video:
        env = VecVideoRecorder(
            env,
            video_folder=str(config.videos_path),
            record_video_trigger=lambda x: x % ((config.total_timesteps * config.n_envs) // config.record_video) == 0,
            video_length=config.record_video_length,
        )
    return env
