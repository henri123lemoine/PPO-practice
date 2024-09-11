from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import config
from environments.simple_humanoid import SimpleHumanoidEnv
from models.ppo_model import create_ppo_model
from utils.callbacks import create_eval_callback


def setup_environment():
    env = SimpleHumanoidEnv()
    env = Monitor(env, config.MONITOR_FILE)
    return DummyVecEnv([lambda: env])


def train_model(env):
    model = create_ppo_model(env, config.TENSORBOARD_LOG)
    eval_callback = create_eval_callback(env, config.BEST_MODEL_PATH, config.LOG_PATH)
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(config.FINAL_MODEL_PATH)
    return model


def evaluate_model(model, env):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=config.EVAL_EPISODES)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def test_model(model, env):
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


def main():
    env = setup_environment()
    model = train_model(env)
    evaluate_model(model, env)
    test_model(model, env)
    env.close()


if __name__ == "__main__":
    main()
