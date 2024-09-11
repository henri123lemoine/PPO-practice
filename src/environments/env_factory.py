from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.config import Config, MONITOR_FILE
from src.environments.simple_humanoid import SimpleHumanoidEnv
from src.environments.mujoco_humanoid import MuJoCoHumanoidEnv


def create_env(config: Config, env_name: str):
    if env_name == "simple_humanoid":
        env = SimpleHumanoidEnv()
        env = Monitor(env, str(MONITOR_FILE))
        env = DummyVecEnv([lambda: env])
    elif env_name == "mujoco_humanoid":
        env = MuJoCoHumanoidEnv()
        env = Monitor(env, str(MONITOR_FILE))
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env
