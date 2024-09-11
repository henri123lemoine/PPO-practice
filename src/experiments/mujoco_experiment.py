from src.train import train
from src.experiments.base_experiment import BaseExperiment


class MujocoExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(
            name="mujoco_humanoid",
            total_timesteps=1_000_000,
            env_name="mujoco_humanoid"
        )


def main():
    experiment = MujocoExperiment()
    config = experiment.get_config()
    train(config, experiment.env_name)


if __name__ == "__main__":
    main()
