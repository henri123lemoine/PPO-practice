# PPO Humanoid Robot Training

This project serves as a practice ground for developing a robust locomotion policy for a humanoid robot using reinforcement learning techniques, especially Proximal Policy Optimization (PPO).

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/henri123lemoine/PPO-practice.git
   cd PPO-practice
   ```

2. Set up a virtual environment:
   ```bash
   uv venv --python 3.12 && source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Copy `.env.example` to `.env` and configure as needed:
   ```bash
   cp .env.example .env
   ```

## Usage

### Running an Experiment

To run an experiment:

```bash
uv run main.py --experiment [experiment_name]
```

Available experiments:

- `experiment_1`: Runs a simple humanoid environment
- `mujoco_humanoid`: Runs the MuJoCo humanoid environment

Example:
```bash
uv run main.py --experiment mujoco_humanoid
```

### Viewing Training Progress

You can view the tensorboard logs by running:

```bash
tensorboard --logdir data/logs/tensorboard
```

Then navigate to `http://localhost:6006` in your browser.

## Future Improvements

1. MuJoCo Enhancements
   - Implement MuJoCo rendering to visualize the training process
   - Set up a system to record videos of the agent's performance at regular intervals (e.g., every X episodes)
   - Develop a user-friendly interface to view and analyze these videos

2. Training Optimizations
   - Implement hyperparameter tuning using techniques like grid search, random search, or Bayesian optimization
   - Explore curriculum learning strategies to gradually increase task difficulty during training
   - Investigate imitation learning techniques to bootstrap the policy with expert demonstrations

3. Domain Randomization
   - Once the Isaac Sim integration is stable, implement domain randomization techniques
   - Randomize physical parameters (mass, friction, etc.), visual aspects, and environmental factors
   - Develop tools to analyze the impact of different randomization strategies on policy robustness

4. Isaac Sim Integration
   - Design the project architecture to facilitate an easy transition from MuJoCo to Isaac Sim
   - Implement an Isaac Sim environment interface compatible with the existing training pipeline

5. Sim2Real Transfer (Future Project)
   - While not part of this project, lay the groundwork for future sim2real transfer
   - Design the training pipeline with real-world deployment in mind
   - Implement logging and analysis tools that can be used to compare sim and real performance

Note: I'll first try to perfect the PPO implementation with MuJoCo before moving on to the Isaac Sim integration.

## Acknowledgments

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for the RL algorithm implementations
- [Gymnasium](https://gymnasium.farama.org/) for the environment interfaces
- [MuJoCo](https://mujoco.org/) for the physics simulation
