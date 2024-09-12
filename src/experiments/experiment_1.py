from src.config import Config
from src.train import train
from src.environments.env_factory import create_env


def main():
    config = Config(
        experiment_name="experiment_1",
        env_name="simple_humanoid",
        n_envs=1,
        total_timesteps=100_000,
        record_video=False,
        train_params={"learning_rate": 1e-3},
    )
    env = create_env(config)
    train(config, env)


if __name__ == "__main__":
    main()
