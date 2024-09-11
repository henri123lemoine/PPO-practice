from abc import ABC, abstractmethod
from typing import Any

from gymnasium.core import RenderFrame
import gymnasium as gym


class BaseEnv(gym.Env, ABC):
    @abstractmethod
    def reset(self, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        pass

    @abstractmethod
    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def seed(self, seed=None) -> tuple[Any, dict[str, Any]]:
        pass
