import gymnasium as gym

from src.environments.base_env import BaseEnv


class MuJoCoHumanoidEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.env = gym.make('Humanoid-v4')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
