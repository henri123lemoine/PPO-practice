import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class SimpleHumanoidEnv(gym.Env):
    def __init__(self):
        super(SimpleHumanoidEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.state = None
        self.steps = 0

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
        

def main():
    # Create environment
    env = SimpleHumanoidEnv()

    # Create and train PPO model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Test the trained model
    obs, _ = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        if done:
            break


if __name__ == "__main__":
    main()
