from src.train import train
from src.environments.env_factory import create_env
from src.experiments.base_experiment import BaseExperiment


class MujocoExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(
            name="mujoco_humanoid",
            total_timesteps=1_000_000,
            env_name="mujoco_humanoid",
            ppo_params={"learning_rate": 3e-4}
        )


def main():
    experiment = MujocoExperiment()
    config = experiment.get_config()
    env = create_env(config, experiment.env_name)
    train(config, env)


if __name__ == "__main__":
    main()
