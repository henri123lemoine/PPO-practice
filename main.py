import logging

from src.experiments.experiment_1 import main as experiment_1
from src.experiments.mujoco_experiment import main as mujoco_experiment

logging.basicConfig(
    level=logging.NOTSET,
    format=f"[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logging.getLogger("matplotlib.font_manager").setLevel(
    logging.ERROR
)  # Deactivate bugged matplotlib DEBUG logger
logger = logging.getLogger(__name__)


def main():
    # Choose which experiment to run
    # experiment_1()
    mujoco_experiment()


if __name__ == "__main__":
    main()
