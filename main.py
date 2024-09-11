import argparse
import logging

from src.experiments.experiment_1 import main as experiment_1
from src.experiments.mujoco_experiment import main as mujoco_experiment

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run PPO experiments")
    parser.add_argument('-e', '--experiment', type=str, default='experiment_1',
                        choices=['experiment_1', 'mujoco_humanoid'],
                        help='Experiment to run (default: experiment_1)')
    args = parser.parse_args()

    if args.experiment == 'experiment_1':
        experiment_1()
    elif args.experiment == 'mujoco_humanoid':
        mujoco_experiment()
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

if __name__ == "__main__":
    main()
