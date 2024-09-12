import logging

from src.config import Config
from src.train import train
from src.environments.env_factory import create_env

logger = logging.getLogger(__name__)


def main():
    config = Config()
    config.update(
        experiment_name="experiment_1",
        env_name="simple_humanoid",
        total_timesteps=500_000,
        record_video=False,
        train_params={
            "learning_rate": 2e-4,
        }
    )

    logger.info(f"Running experiment: {config.experiment_name}")
    logger.debug(f"Config: {config}")

    env = create_env(config)
    train(config, env)


if __name__ == "__main__":
    main()
