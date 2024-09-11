from dataclasses import dataclass, field

from src.config import Config

@dataclass
class BaseExperiment:
    name: str
    total_timesteps: int = 100_000
    eval_episodes: int = 10
    ppo_params: dict[str, int | float] = field(default_factory=lambda: {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    })
    env_name: str = "simple_humanoid"

    def get_config(self) -> Config:
        return Config(
            experiment_name=self.name,
            total_timesteps=self.total_timesteps,
            eval_episodes=self.eval_episodes,
            ppo_params=self.ppo_params
        )
