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
        total_timesteps=2_000_000,
        record_video=True,
        train_params={
            "learning_rate": 2e-4,
            "ent_coef": 0.01,
        }
    )

    logger.info(f"Running experiment: {config.experiment_name}")
    logger.debug(f"Config: {config}")

    env = create_env(config)
    train(config, env)


if __name__ == "__main__":
    main()
