import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_vec_env

from src.config import Config
from src.environments.simple_humanoid import SimpleHumanoidEnv
from src.environments.mujoco_humanoid import MuJoCoHumanoidEnv


def create_env(config: Config, env_name: str):
    if env_name == "simple_humanoid":
        env = SimpleHumanoidEnv()
    elif env_name == "mujoco_humanoid":
        # env = MuJoCoHumanoidEnv()  # this didn't work because it doesn't behave sufficiently like any other gym environments
        env = gym.make("Humanoid-v4", render_mode="rgb_array")
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    env = make_vec_env(
        env_id = lambda: env, env_kwargs = None,

        n_envs = 1,
        seed = config.seed,
        start_index = 0,

        monitor_dir = str(config.experiment_path), monitor_kwargs=None,

        wrapper_class = RecordVideo,
        wrapper_kwargs={
            "video_folder": str(config.videos_path),
            "episode_trigger": lambda step_id: step_id % 100 == 0,
            "step_trigger": None,
            "video_length": 0,
            "name_prefix": "rl-video",
            "disable_logger": False
        },

        # A custom ``VecEnv`` class constructor, and kwargs
        vec_env_cls = None, vec_env_kwargs = None,
    )
    return env
