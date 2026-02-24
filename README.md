# Policy Gradient Methods for Inverted Pendulum Control

Course project for Reinforcement Learning.
Author: Nikolija Cuckić

## Overview

This project implements and compares two policy gradient algorithms:

- REINFORCE (vanilla Monte Carlo policy gradient)
- REINFORCE with Value Baseline

Both algorithms are evaluated on continuous-action MuJoCo environments:
`InvertedPendulum-v5` and `InvertedDoublePendulum-v5`.

Additional experiments cover transfer learning from the single to the double
pendulum, and robustness evaluation under external wind perturbations.

## Project Structure

```
rl_pendulum_project/
  src/
    agent_vanilla.py          Vanilla REINFORCE agent
    agent_baseline.py         REINFORCE with value baseline agent
    networks.py               Policy and value network definitions
    utils.py                  Seed setting and CSV logging utilities
  training/
    main_train_vanilla.py     Train vanilla REINFORCE on InvertedPendulum-v5
    main_train_baseline.py    Train REINFORCE+Baseline on InvertedPendulum-v5
    main_train_double_pendulum.py  Train on InvertedDoublePendulum-v5
    main_transfer_learning.py Transfer weights from single to double pendulum
  scripts/
    generate_plots.py         Generate all analysis plots from CSV logs
    generate_videos.py        Generate all evaluation MP4 videos
    robustness_test.py        Evaluate agents under wind force perturbations
  checkpoints/                Saved model weights (.pth)
  logs/                       Training reward logs (.csv)
  plots/                      Generated analysis plots (.png)
  videos/                     Generated evaluation videos (.mp4)
  izvestaj.tex                LaTeX report (Serbian, Latin script)
  requirements.txt            Python dependencies
```

> Note: all scripts must be run from the project root directory so that
> relative paths to `checkpoints/`, `logs/`, `src/` resolve correctly.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Running Experiments

Train on InvertedPendulum-v5 (3 seeds each, 5000 episodes):

```bash
python training/main_train_vanilla.py
python training/main_train_baseline.py
```

Train on InvertedDoublePendulum-v5 (seed 3, 10000 episodes):

```bash
python training/main_train_double_pendulum.py
```

Transfer learning (baseline seed 3 as source, 10000 fine-tuning episodes):

```bash
python training/main_transfer_learning.py
```

## Generating Outputs

```bash
python scripts/generate_plots.py      # saves 10 plots to plots/
python scripts/generate_videos.py     # saves 5 comparison videos to videos/
python scripts/robustness_test.py     # saves robustness plots and 4 videos to videos/
```

## Algorithms

### REINFORCE

Policy update per timestep within an episode:

    theta <- theta + alpha * G_t * grad log pi(a_t | s_t, theta)

where G_t is the Monte Carlo return from step t.

### REINFORCE with Baseline

    theta <- theta + alpha * (G_t - V(s_t)) * grad log pi(a_t | s_t, theta)

V(s_t) is a learned value network trained to minimize (G_t - V(s_t))^2.
The baseline reduces gradient variance without introducing bias.

### Policy Representation

Actions are sampled from a Gaussian policy:

    pi(a | s, theta) = N(mu(s, theta), sigma(s, theta)^2)

The policy network outputs both mean and standard deviation.

## Key Findings

- REINFORCE with baseline converges faster and with lower variance than
  vanilla REINFORCE across all seeds.
- Both algorithms reliably solve InvertedPendulum-v5 (max reward 1000).
- InvertedDoublePendulum-v5 remains largely unsolved due to the high variance
  inherent in Monte Carlo policy gradient methods on long-horizon tasks.
- Transfer learning provides a modest improvement in early episodes but does
  not significantly change asymptotic performance.
- Robustness to wind perturbations degrades with increasing force, though
  non-monotonic behavior was observed at moderate perturbation levels.

## Dependencies

- Python 3.10+
- PyTorch
- Gymnasium (with MuJoCo)
- NumPy, Pandas, Matplotlib, Seaborn
- imageio[ffmpeg], Pillow
