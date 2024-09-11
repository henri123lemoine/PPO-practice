from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import PolicyPredictor
from stable_baselines3.common.callbacks import EvalCallback

from src.config import Config, MONITOR_FILE, TENSORBOARD_LOG, BEST_MODEL_PATH, LOG_PATH, FINAL_MODEL_PATH
from src.environments.env_factory import create_env
from src.models.ppo_model import create_ppo_model


def train_model(env, config: Config) -> PPO:
    model = create_ppo_model(env, TENSORBOARD_LOG, config.PPO_PARAMS)
    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=BEST_MODEL_PATH,
        log_path=LOG_PATH,
        eval_freq=2000,
        deterministic=True,
        render=False
    )
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(FINAL_MODEL_PATH)
    return model


def evaluate_model(model: PolicyPredictor, env, config: Config):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=config.EVAL_EPISODES)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def test_model(model: PolicyPredictor, env, config: Config):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        if dones:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]


def train(config: Config, env_name: str):
    env = create_env(config, env_name)
    model = train_model(env, config)
    evaluate_model(model, env, config)
    test_model(model, env, config)
    env.close()
