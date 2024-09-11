from src.train import train
from src.environments.env_factory import create_env
from src.experiments.base_experiment import BaseExperiment


class Experiment1(BaseExperiment):
    def __init__(self):
        super().__init__(
            name="experiment_1",
            total_timesteps=100_000,
            ppo_params={"learning_rate": 1e-4}  # Override default learning rate
        )


def main():
    experiment = Experiment1()
    config = experiment.get_config()
    env = create_env(config, experiment.env_name)
    train(config, env)


if __name__ == "__main__":
    main()
