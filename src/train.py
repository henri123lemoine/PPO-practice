from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import PolicyPredictor
from stable_baselines3.common.callbacks import EvalCallback

from src import config
from src.environments.simple_humanoid import SimpleHumanoidEnv
from src.models.ppo_model import create_ppo_model


def setup_environment():
    env = SimpleHumanoidEnv()
    env = Monitor(env, str(config.MONITOR_FILE))
    return DummyVecEnv([lambda: env])


def train_model(env) -> PPO:
    model = create_ppo_model(env, config.TENSORBOARD_LOG)
    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=config.BEST_MODEL_PATH,
        log_path=config.LOG_PATH,
        eval_freq=2000,
        deterministic=True,
        render=False
    )
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(config.FINAL_MODEL_PATH)
    return model


def evaluate_model(model: PolicyPredictor, env):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=config.EVAL_EPISODES)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def test_model(model: PolicyPredictor, env):
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
