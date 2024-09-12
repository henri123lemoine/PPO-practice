import logging

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import PolicyPredictor
from stable_baselines3.common.vec_env import VecEnv

from src.config import Config
from src.models.ppo_model import create_ppo_model

logger = logging.getLogger(__name__)


def train_model(env, config: Config) -> PPO:
    model = create_ppo_model(env, config)
    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=config.best_model_path,
        log_path=config.tensorboard_log_path,
        eval_freq=2000,
        deterministic=True,
        render=False
    )
    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)
    model.save(config.final_model_path)
    return model


def evaluate_model(model: PolicyPredictor, env: gym.Env, config: Config):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=config.eval_episodes)
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def test_model(model: PolicyPredictor, env: gym.Env, config: Config):
    if isinstance(env, VecEnv):
        obs = env.reset()
    else:
        obs, _ = env.reset()
    
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(env, VecEnv):
            obs, _, done, _ = env.step(action)
            terminated = done[0]
            truncated = False  # VecEnv doesn't provide truncated information
        else:
            obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            if isinstance(env, VecEnv):
                obs = env.reset()
            else:
                obs, _ = env.reset()


def train(config: Config, env: gym.Env):
    model = train_model(env, config)
    evaluate_model(model, env, config)
    test_model(model, env, config)
    env.close()
