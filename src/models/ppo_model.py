from stable_baselines3 import PPO


def create_ppo_model(env, tensorboard_log):
    return PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
