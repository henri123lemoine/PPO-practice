import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# General
DATE = datetime.now().strftime("%Y-%m-%d")
PROJECT_PATH = PROJECT_DIR = Path(__file__).resolve().parent
IS_DEVELOPMENT = os.environ.get("ENVIRONMENT", "development").lower() in ("development", "dev", "d")

# Data
DATA_PATH = PROJECT_PATH / "data"
MISC_PATH = DATA_PATH / "misc"
CACHE_PATH = DATA_PATH / ".cache"
PLOTS_PATH = DATA_PATH / "plots"
LOG_PATH = DATA_PATH / "logs"
[os.makedirs(p, exist_ok=True) for p in (DATA_PATH, MISC_PATH, CACHE_PATH, PLOTS_PATH, LOG_PATH)]
