from src.train import train
from src.config import Config


def main():
    Config.EXPERIMENT_NAME = "experiment_1"
    Config.TOTAL_TIMESTEPS = 100_000
    Config.EVAL_EPISODES = 10

    Config.PPO_PARAMS = {
        **Config.PPO_PARAMS,
        "learning_rate": 1e-4,  # Override the default learning rate
    }

    train(Config)


if __name__ == "__main__":
    main()
