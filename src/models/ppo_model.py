from stable_baselines3 import PPO

from src.config import Config


def create_ppo_model(env, config: Config) -> PPO:
    """
    Create a PPO model with the given environment and parameters.

    Args:
        env (gym.Env): The environment to train on.
        config (Config): The configuration to use for training.

    Returns:
        PPO: The created PPO model.
    """
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=config.tensorboard_path,
        **config.train_params
    )
