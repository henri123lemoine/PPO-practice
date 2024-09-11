from stable_baselines3 import PPO


def create_ppo_model(env, tensorboard_log: str, ppo_params: dict[str, int | float]) -> PPO:
    return PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, **ppo_params)
