from src.experiments.experiment_1 import main as experiment_1
from src.experiments.mujoco_experiment import main as mujoco_experiment

def main():
    # Choose which experiment to run
    # experiment_1()
    mujoco_experiment()

if __name__ == "__main__":
    main()
