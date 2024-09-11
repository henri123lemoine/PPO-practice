from stable_baselines3 import PPO


def create_ppo_model(env, tensorboard_log: str, ppo_params: dict[str, int | float]) -> PPO:
    """
    Create a PPO model with the given environment and parameters.

    Args:
        env (gym.Env): The environment to train on.
        tensorboard_log (str): Path to save tensorboard logs.
        ppo_params (dict): Parameters for the PPO algorithm.

    Returns:
        PPO: The created PPO model.
    """
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        **ppo_params
    )
