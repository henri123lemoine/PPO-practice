import os
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


# General
DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PROJECT_PATH = Path(__file__).resolve().parent.parent
IS_DEVELOPMENT = os.environ.get("ENVIRONMENT", "development").lower() in ("development", "dev", "d")

# Data
DATA_PATH = PROJECT_PATH / "data"
CACHE_PATH = DATA_PATH / ".cache"
MODEL_PATH = DATA_PATH / "models"
# Specific model path is determined by the experiment name and the date and time. E.g.:
# "models/mujoco_humanoid/2021-08-14_14-23-00/"
os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

@dataclass
class Config:
    experiment_name: str = "unconfigured"

    n_envs: int = 1
    seed: int = 42
    total_timesteps: int = 100_000
    eval_episodes: int = 10

    train_params: dict[str, int | float] = field(default_factory=lambda: {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    })

    @property
    def experiment_path(self) -> Path:
        return MODEL_PATH / f"{self.experiment_name}_{DATE_TIME}"

    @property
    def tensorboard_log_path(self) -> Path:
        path = self.experiment_path / "tensorboard"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def best_model_path(self) -> Path:
        path = self.experiment_path / "best_model"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def final_model_path(self) -> Path:
        path = self.experiment_path / "final_model"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def videos_path(self) -> Path:
        path = self.experiment_path / "videos"
        path.mkdir(parents=True, exist_ok=True)
        return path
