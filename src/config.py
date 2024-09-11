import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


# General
DATE = datetime.now().strftime("%Y-%m-%d")
PROJECT_PATH = Path(__file__).resolve().parent.parent
IS_DEVELOPMENT = os.environ.get("ENVIRONMENT", "development").lower() in ("development", "dev", "d")

# Data
DATA_PATH = PROJECT_PATH / "data"
LOG_PATH = DATA_PATH / "logs"
MODEL_PATH = DATA_PATH / "models"
CACHE_PATH = DATA_PATH / ".cache"
PLOTS_PATH = DATA_PATH / "plots"

# Training
## Paths
TENSORBOARD_LOG = LOG_PATH / "tensorboard"
MONITOR_FILE = LOG_PATH / "monitor.csv"
BEST_MODEL_PATH = MODEL_PATH / "best_model"
FINAL_MODEL_PATH = MODEL_PATH / "final_model"

# Create directories
for p in (DATA_PATH, LOG_PATH, MODEL_PATH, CACHE_PATH, PLOTS_PATH):
    if not p.exists():
        os.makedirs(p, exist_ok=True)

@dataclass
class Config:
    EXPERIMENT_NAME: str = "unconfigured"

    ## Default Hyperparameters
    PPO_PARAMS: dict[str, int | float] = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    }

    TOTAL_TIMESTEPS: int = 100_000
    EVAL_EPISODES: int = 10
