# PPO Humanoid Robot Training

This project implements a Proximal Policy Optimization (PPO) algorithm to train a humanoid robot, with the ultimate goal of using it in an Isaac Sim environment.

## Setup

1. Clone the repository:
   ```bash
   git clone https://henri123lemoine/PPO-practice.git
   cd your-project-name
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
uv run main.py
```

This will execute the experiment defined in `experiments/experiment_1.py`.

## Future Work

- Integration with Isaac Sim for more realistic humanoid robot simulation.
- Implementation of advanced logging and visualization tools.
