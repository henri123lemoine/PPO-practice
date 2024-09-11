from typing import Any

import gymnasium as gym

from src.environments.base_env import BaseEnv


class MuJoCoHumanoidEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.env = gym.make('Humanoid-v4')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def seed(self, seed=None) -> tuple[Any, dict[str, Any]]:
        return self.env.reset(seed=seed)
