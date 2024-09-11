from abc import ABC, abstractmethod

import gymnasium as gym


class BaseEnv(gym.Env, ABC):
    @abstractmethod
    def reset(self, seed=None, options=None):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def seed(self, seed=None):
        pass
