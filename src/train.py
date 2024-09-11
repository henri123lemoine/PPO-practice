import logging

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import PolicyPredictor
from stable_baselines3.common.callbacks import EvalCallback

from src.config import Config, TENSORBOARD_LOG, BEST_MODEL_PATH, LOG_PATH, FINAL_MODEL_PATH
from src.models.ppo_model import create_ppo_model

logger = logging.getLogger(__name__)


def train_model(env, config: Config) -> PPO:
    model = create_ppo_model(env, TENSORBOARD_LOG, config.ppo_params)
    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=BEST_MODEL_PATH,
        log_path=LOG_PATH,
        eval_freq=2000,
        deterministic=True,
        render=False
    )
    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)
    model.save(FINAL_MODEL_PATH)
    return model


def evaluate_model(model: PolicyPredictor, env: gym.Env, config: Config):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=config.eval_episodes)
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def test_model(model: PolicyPredictor, env: gym.Env, config: Config):
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


def train(config: Config, env: gym.Env):
    model = train_model(env, config)
    evaluate_model(model, env, config)
    test_model(model, env, config)
    env.close()
