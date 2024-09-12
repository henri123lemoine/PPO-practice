import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.config import Config


def make_env(config: Config):

    if config.env_name == "simple_humanoid":
        from src.environments.simple_humanoid import SimpleHumanoidEnv
        env = SimpleHumanoidEnv()
    elif config.env_name == "mujoco_humanoid":
        from src.environments.mujoco_humanoid import MuJoCoHumanoidEnv
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
            episode_trigger=config.episode_trigger,
            step_trigger=None,
            video_length=0,
            name_prefix="rl-video",
            disable_logger=False,
        )

    return env

def create_env(config: Config):
    env = make_vec_env(
        env_id=lambda: make_env(config),
        n_envs=config.n_envs,
        seed=config.seed,
    )
    return env
