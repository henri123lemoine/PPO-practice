from typing import Any

import numpy as np
from gymnasium import spaces

from src.environments.base_env import BaseEnv


class SimpleHumanoidEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.state = None
        self.steps = 0
        self.render_mode = None

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(8,))
        self.steps = 0
        return self.state, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.state += action.repeat(2) * 0.1
        self.steps += 1
        reward = -np.sum(np.square(self.state))
        terminated = self.steps >= 100
        truncated = False  # Not used
        return self.state, reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        return np.zeros((300, 400, 3), dtype=np.uint8)

    def close(self) -> None:
        pass

    def seed(self, seed=None) -> list[int]:
        return super().reset(seed=seed)
