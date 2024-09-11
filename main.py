import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from settings import LOG_PATH


class SimpleHumanoidEnv(gym.Env):
    def __init__(self):
        super(SimpleHumanoidEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.state = None
        self.steps = 0
        self.render_mode = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(8,))
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.state += action.repeat(2) * 0.1
        self.steps += 1
        reward = -np.sum(np.square(self.state))  # Simple reward for being close to origin
        done = self.steps >= 100
        return self.state, reward, done, False, {}

    def render(self):
        return np.zeros((300, 400, 3), dtype=np.uint8)  # blank image


# Create directories
model_dir = os.path.join(LOG_PATH, "models")
os.makedirs(model_dir, exist_ok=True)

# Create and wrap the environment
env = SimpleHumanoidEnv()
env = Monitor(env, os.path.join(LOG_PATH, "monitor.csv"))
env = DummyVecEnv([lambda: env])

# Set up callbacks
eval_callback = EvalCallback(env, 
                             best_model_save_path=os.path.join(model_dir, "best_model"),
                             log_path=os.path.join(LOG_PATH, "results"), 
                             eval_freq=2000,
                             deterministic=True, 
                             render=False)

# Create and train PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(LOG_PATH, "tensorboard"))
model.learn(total_timesteps=100000, callback=eval_callback)

# Save the final model
model.save(os.path.join(model_dir, "final_model"))

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test the trained model
obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    if dones:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

env.close()
