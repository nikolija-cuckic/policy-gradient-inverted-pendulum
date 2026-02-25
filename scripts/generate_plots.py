"""
generate_plots.py
-----------------
Generates all analysis plots from CSV logs and saves them to plots/.

Plots produced:
  01_vanilla_learning_curves.png      - per-seed reward curves for Vanilla REINFORCE
  02_baseline_learning_curves.png     - per-seed reward curves for REINFORCE+baseline
  03_vanilla_vs_baseline.png          - mean ± std comparison of both algorithms
  04_variance_comparison.png          - variability (std) across seeds, by algorithm
  05_final_performance_bar.png        - bar chart: mean reward in last 500 ep per model
  06_double_pendulum_curve.png        - training curve for double pendulum (from scratch)
  07_transfer_vs_scratch.png          - transfer learning vs scratch on double pendulum
  08_all_seeds_grid.png               - 2x3 grid: all 6 seed curves side by side
  09_reward_distribution.png          - violin plot of reward distribution per model
  10_convergence_speed.png            - episodes to reach 90% of max reward
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns

#  style
sns.set_theme(style="darkgrid", palette="tab10")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LOGS_DIR  = "logs"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

WINDOW = 100   # smoothing window (episodes)


#  I/O helpers 

def load_csv(filename: str) -> pd.Series:
    path = os.path.join(LOGS_DIR, filename)
    df = pd.read_csv(path)
    return df["reward"].reset_index(drop=True)


def smooth(series: pd.Series, window: int = WINDOW) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def load_seeds(prefix: str, seeds=(1, 2, 3)) -> list[pd.Series]:
    """Load reward series for multiple seeds. Skips missing files."""
    result = []
    for s in seeds:
        fname = f"{prefix}_seed{s}.csv"
        fpath = os.path.join(LOGS_DIR, fname)
        if os.path.exists(fpath):
            result.append(load_csv(fname))
        else:
            print(f"  [SKIP] {fname} not found")
    return result


def mean_std_band(series_list: list[pd.Series]):
    """Compute aligned mean and std across multiple series (trimmed to shortest)."""
    min_len = min(len(s) for s in series_list)
    arr = np.array([s.values[:min_len] for s in series_list])
    return arr.mean(axis=0), arr.std(axis=0), min_len


def save_fig(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# vanilla learning curves

def plot_vanilla_seeds():
    series = load_seeds("vanilla")
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Blues_d", len(series))
    for i, (s, c) in enumerate(zip(series, colors), start=1):
        ax.plot(smooth(s), label=f"Seed {i}", color=c, linewidth=1.8)
    ax.set_title("Vanilla REINFORCE - Learning Curves (3 seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (smoothed, w={WINDOW})")
    ax.axhline(1000, color="black", linestyle="--", linewidth=1, label="Max reward (1000)")
    ax.legend()
    save_fig(fig, "01_vanilla_learning_curves.png")


# baseline learning curves 

def plot_baseline_seeds():
    series = load_seeds("baseline")
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Greens_d", len(series))
    for i, (s, c) in enumerate(zip(series, colors), start=1):
        ax.plot(smooth(s), label=f"Seed {i}", color=c, linewidth=1.8)
    ax.set_title("REINFORCE + Baseline - Learning Curves (3 seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (smoothed, w={WINDOW})")
    ax.axhline(1000, color="black", linestyle="--", linewidth=1, label="Max reward (1000)")
    ax.legend()
    save_fig(fig, "02_baseline_learning_curves.png")


# vanilla vs baseline

def plot_vanilla_vs_baseline():
    v_series = load_seeds("vanilla")
    b_series = load_seeds("baseline")
    if not v_series or not b_series:
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    for label, series_list, color in [
        ("Vanilla REINFORCE", v_series, "#3274A1"),
        ("REINFORCE + Baseline", b_series, "#2CA02C"),
    ]:
        mean, std, n = mean_std_band([smooth(s) for s in series_list])
        x = np.arange(n)
        ax.plot(x, mean, label=label, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.axhline(1000, color="black", linestyle="--", linewidth=1, label="Max reward")
    ax.set_title("Vanilla REINFORCE vs. REINFORCE + Baseline\n(Mean ± 1 Std across 3 seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (smoothed, w={WINDOW})")
    ax.legend()
    save_fig(fig, "03_vanilla_vs_baseline.png")


# variance comparison (rolling std)

def plot_variance_comparison():
    v_series = load_seeds("vanilla")
    b_series = load_seeds("baseline")
    if not v_series or not b_series:
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    for label, series_list, color in [
        ("Vanilla REINFORCE", v_series, "#3274A1"),
        ("REINFORCE + Baseline", b_series, "#2CA02C"),
    ]:
        _, std, n = mean_std_band(series_list)
        std_smooth = pd.Series(std).rolling(WINDOW, min_periods=1).mean()
        ax.plot(np.arange(n), std_smooth, label=label, linewidth=2, color=color)

    ax.set_title("Training Variance (Std across seeds)\nLower = more stable")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Standard Deviation of Reward")
    ax.legend()
    save_fig(fig, "04_variance_comparison.png")


#  final performance chart 

def plot_final_performance(last_n: int = 500):
    entries = [
        ("Vanilla\nSeed 1", "vanilla_seed1.csv"),
        ("Vanilla\nSeed 2", "vanilla_seed2.csv"),
        ("Vanilla\nSeed 3", "vanilla_seed3.csv"),
        ("Baseline\nSeed 1", "baseline_seed1.csv"),
        ("Baseline\nSeed 2", "baseline_seed2.csv"),
        ("Baseline\nSeed 3", "baseline_seed3.csv"),
    ]
    labels, means, stds = [], [], []
    for label, fname in entries:
        fpath = os.path.join(LOGS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        s = load_csv(fname)
        tail = s.iloc[-last_n:]
        labels.append(label)
        means.append(tail.mean())
        stds.append(tail.std())

    if not labels:
        return

    colors = ["#3274A1"] * 3 + ["#2CA02C"] * 3
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(labels)],
                  edgecolor="white", linewidth=0.8, width=0.6)
    ax.axhline(1000, color="black", linestyle="--", linewidth=1, label="Max reward")

    # value annotations
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{m:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Final Performance (mean of last {last_n} episodes)")
    ax.set_ylim(0, 1150)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#3274A1", label="Vanilla REINFORCE"),
                       Patch(facecolor="#2CA02C", label="REINFORCE + Baseline")]
    ax.legend(handles=legend_elements)
    save_fig(fig, "05_final_performance_bar.png")


# Double Pendulum training curve

def plot_double_pendulum():
    fpath = os.path.join(LOGS_DIR, "double_pendulum_seed3.csv")
    if not os.path.exists(fpath):
        print("  [SKIP] double_pendulum_seed3.csv not found")
        return
    s = load_csv("double_pendulum_seed3.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(smooth(s).values, color="#9467BD", linewidth=2.2, label=f"Smoothed (w={WINDOW})")
    ax.set_title("InvertedDoublePendulum-v5 - Training from Scratch")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    save_fig(fig, "06_double_pendulum_curve.png")


# transfer Learning vs Scratch

def plot_transfer_vs_scratch():
    scratch_path   = os.path.join(LOGS_DIR, "double_pendulum_seed3.csv")
    transfer_path  = os.path.join(LOGS_DIR, "transfer_learning_seed3.csv")

    if not os.path.exists(scratch_path) and not os.path.exists(transfer_path):
        print("  [SKIP] no double pendulum or transfer learning CSV found")
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    for fname, label, color in [
        ("double_pendulum_seed3.csv",   "Scratch - seed3 (InvertedDoublePendulum)", "#9467BD"),
        ("transfer_learning_seed3.csv", "Transfer Learning (from SP → DP)",         "#D62728"),
    ]:
        fpath = os.path.join(LOGS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        s = load_csv(fname)
        ax.plot(smooth(s).values, label=label, color=color, linewidth=2.2)

    ax.set_title("Transfer Learning vs. Training from Scratch\n(InvertedDoublePendulum-v5)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Episode Reward (smoothed, w={WINDOW})")
    ax.legend()
    save_fig(fig, "07_transfer_vs_scratch.png")


# 2x3 seed grid 

def plot_all_seeds_grid():
    v_series = load_seeds("vanilla")
    b_series = load_seeds("baseline")
    if not v_series or not b_series:
        return
    all_series = v_series + b_series
    titles = [f"Vanilla Seed {i}" for i in range(1, len(v_series)+1)] + \
             [f"Baseline Seed {i}" for i in range(1, len(b_series)+1)]

    n = len(all_series)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows), sharey=True)
    axes = axes.flatten()

    colors = ["#3274A1"]*len(v_series) + ["#2CA02C"]*len(b_series)

    for ax, s, title, color in zip(axes, all_series, titles, colors):
        ax.plot(smooth(s).values, color=color, linewidth=2)
        ax.axhline(1000, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.xaxis.set_major_locator(MaxNLocator(5))

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("All Training Runs (Vanilla vs Baseline, per seed)", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "08_all_seeds_grid.png")


# reward distribution

def plot_reward_distribution():
    entries = {
        "Vanilla S1": "vanilla_seed1.csv",
        "Vanilla S2": "vanilla_seed2.csv",
        "Vanilla S3": "vanilla_seed3.csv",
        "Baseline S1": "baseline_seed1.csv",
        "Baseline S2": "baseline_seed2.csv",
        "Baseline S3": "baseline_seed3.csv",
    }
    data, labels = [], []
    for label, fname in entries.items():
        fpath = os.path.join(LOGS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        s = load_csv(fname)
        data.append(s.values)
        labels.append(label)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = (["#3274A1"]*3 + ["#2CA02C"]*3)[:len(data)]
    parts = ax.violinplot(data, showmedians=True, showextrema=True)

    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward Distribution per Run (Violin Plot)")
    ax.axhline(1000, color="black", linestyle="--", linewidth=1, label="Max reward")
    ax.legend()
    save_fig(fig, "09_reward_distribution.png")


# convergence speed

def plot_convergence_speed(threshold: float = 0.9):
    """Episode at which smoothed reward first reaches threshold * 1000."""
    entries = {
        "Vanilla S1":  "vanilla_seed1.csv",
        "Vanilla S2":  "vanilla_seed2.csv",
        "Vanilla S3":  "vanilla_seed3.csv",
        "Baseline S1": "baseline_seed1.csv",
        "Baseline S2": "baseline_seed2.csv",
        "Baseline S3": "baseline_seed3.csv",
    }
    target = threshold * 1000
    labels, convergence = [], []
    for label, fname in entries.items():
        fpath = os.path.join(LOGS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        s = smooth(load_csv(fname))
        reached = np.where(s.values >= target)[0]
        conv = int(reached[0]) if len(reached) > 0 else len(s)
        labels.append(label)
        convergence.append(conv)

    if not labels:
        return

    colors = (["#3274A1"]*3 + ["#2CA02C"]*3)[:len(labels)]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, convergence, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

    for bar, v in zip(bars, convergence):
        label_text = str(v) if v < 5000 else "Never"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                label_text, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Episode")
    ax.set_title(f"Convergence Speed\n(first episode with smoothed reward ≥ {int(target)})")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#3274A1", label="Vanilla REINFORCE"),
                       Patch(facecolor="#2CA02C", label="REINFORCE + Baseline")]
    ax.legend(handles=legend_elements)
    save_fig(fig, "10_convergence_speed.png")


# main 

if __name__ == "__main__":
    print("Generating plots from logs/ ...\n")

    plot_vanilla_seeds()
    plot_baseline_seeds()
    plot_vanilla_vs_baseline()
    plot_variance_comparison()
    plot_final_performance()
    plot_double_pendulum()
    plot_transfer_vs_scratch()
    plot_all_seeds_grid()
    plot_reward_distribution()
    plot_convergence_speed()

    print(f"\nDone! All plots saved to: {PLOTS_DIR}/")
