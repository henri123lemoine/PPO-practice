import logging

from src.config import Config
from src.train import train
from src.environments.env_factory import create_env

logger = logging.getLogger(__name__)


def main():
    config = Config()
    config.update(
        experiment_name="mujoco_humanoid",
        env_name="Humanoid-v4",
        total_timesteps=20_000_000,
        record_video=True,
        record_video_freq=20,
    )

    logger.info(f"Running experiment: {config.experiment_name}")
    logger.debug(f"Config: {config}")

    env = create_env(config)
    train(config, env)


if __name__ == "__main__":
    main()
