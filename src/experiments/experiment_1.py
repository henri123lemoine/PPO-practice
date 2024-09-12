from src.config import Config
from src.train import train
from src.environments.env_factory import create_env


def main():
    config = Config(
        experiment_name="experiment_1",
        total_timesteps=100_000,
        env_name="simple_humanoid",
        train_params={"learning_rate": 1e-4},
        record_video=False,
    )
    env = create_env(config)
    train(config, env)


if __name__ == "__main__":
    main()
