from src.config import Config
from src.train import train
from src.environments.env_factory import create_env


def main():
    n_envs = 8
    config = Config(
        experiment_name="mujoco_humanoid",
        env_name="mujoco_humanoid",

        n_envs=n_envs,
        total_timesteps=1_000_000,

        monitor=True,
        record_video=True,
        episode_trigger=lambda step: step % 100 == 0,

        train_params={
            "learning_rate": 3e-4,
            "n_steps": 2048 // n_envs,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
    )
    env = create_env(config)
    train(config, env)


if __name__ == "__main__":
    main()
