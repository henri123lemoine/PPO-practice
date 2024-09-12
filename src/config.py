import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# General
DATE_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PROJECT_PATH = Path(__file__).resolve().parent.parent
IS_DEVELOPMENT = os.environ.get("ENVIRONMENT", "development").lower() in ("development", "dev", "d")

# Data
DATA_PATH = PROJECT_PATH / "data"
CACHE_PATH = DATA_PATH / ".cache"
EXPERIMENTS_PATH = DATA_PATH / "experiments"

for path in [CACHE_PATH, EXPERIMENTS_PATH]:
    os.makedirs(path, exist_ok=True)


@dataclass
class Config:
    experiment_name: str = "not-configured"
    env_name: str = "not-configured"
    verbose: int = 0

    n_envs: int = 4
    seed: int = 42
    total_timesteps: int = 1_000_000

    # Evaluation settings
    eval_episodes: int = 10
    eval_freq: int = 50_000

    # Video recording settings
    record_video: bool = True
    record_video_freq: int = 100_000
    record_video_length: int = 200
    render_mode: str = "rgb_array"

    train_params: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
    })

    def __post_init__(self):
        self.train_params["n_steps"] = self.train_params["n_steps"] // self.n_envs

    @property
    def record_video_trigger_interval(self) -> int:
        num_steps_per_ep = 100
        trigger_interval = self.total_timesteps // self.record_video_freq
        return trigger_interval

    @property
    def record_video_trigger(self) -> Callable[[int], bool]:
        logger.info(f"Setting up video recording. Trigger interval: {self.record_video_trigger_interval}")
        def debug_trigger(x):
            # Divide x by n_envs to get the actual global step count
            global_step = x // self.n_envs
            should_record = global_step % self.record_video_trigger_interval == 0
            if should_record:
                logger.debug(f"Recording video at step {x}")
            return should_record
        return debug_trigger

    @property
    def experiment_path(self) -> Path:
        return EXPERIMENTS_PATH / self.experiment_name

    @property
    def run_path(self) -> Path:
        return self.experiment_path / f"run_{DATE_TIME}"

    @property
    def tensorboard_path(self) -> Path:
        return self.experiment_path / "tensorboard"

    @property
    def best_model_path(self) -> Path:
        return self.run_path / "best_model"

    @property
    def final_model_path(self) -> Path:
        return self.run_path / "final_model"

    @property
    def videos_path(self) -> Path:
        return self.run_path / "videos"

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key == "train_params":
                self.train_params.update(value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
