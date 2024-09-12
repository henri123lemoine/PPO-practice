from src.config import Config
from src.train import train
from src.environments.env_factory import create_env


def main():
    config = Config(
        experiment_name="mujoco_humanoid",
        env_name="mujoco_humanoid",
        total_timesteps=1_000_000,
        n_envs=1,
        train_params={"learning_rate": 3e-4},
    )
    env = create_env(config)
    train(config, env)


if __name__ == "__main__":
    main()
