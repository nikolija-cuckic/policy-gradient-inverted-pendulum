"""
robustness_test.py
------------------
Tests robustness of trained agents against external horizontal wind forces.

Generates:
  Video (single pendulum):
    - robustness_vanilla.mp4      : Vanilla REINFORCE, 4 panels (wind 0,2,4,8 N)
    - robustness_baseline.mp4     : REINFORCE+Baseline, 4 panels
  Video (double pendulum):
    - robustness_dp_scratch.mp4   : Scratch seed3, 4 panels
    - robustness_dp_transfer.mp4  : Transfer seed3, 4 panels
  Plots:
    - 09_robustness_single_pendulum.png
    - 10_robustness_double_pendulum.png
"""

import os
import numpy as np
import torch
import gymnasium as gym
import imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns

from src.agent_vanilla import VanillaREINFORCE
from src.agent_baseline import REINFORCEWithBaseline

# configuration

WIND_FORCES    = [0, 2, 4, 8]   # N applied horizontally
N_EVAL_EPS     = 5              # episodes per wind level
VIDEOS_DIR     = "videos"
PLOTS_DIR      = "plots"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_style("whitegrid")


# load helpers 

def load_vanilla(path: str, env_name: str) -> VanillaREINFORCE:
    env = gym.make(env_name)
    agent = VanillaREINFORCE(env.observation_space.shape[0], env.action_space.shape[0])
    agent.policy_net.load_state_dict(torch.load(path, map_location="cpu"))
    env.close()
    return agent


def load_baseline(path: str, env_name: str) -> REINFORCEWithBaseline:
    env = gym.make(env_name)
    agent = REINFORCEWithBaseline(env.observation_space.shape[0], env.action_space.shape[0])
    ckpt = torch.load(path, map_location="cpu")
    agent.policy_net.load_state_dict(ckpt["policy"])
    agent.value_net.load_state_dict(ckpt["value"])
    env.close()
    return agent


# wind episode helpers

def record_with_wind(agent, env_name: str, wind: float,
                     max_steps: int = 500, seed: int = 42):
    """Record one episode with constant horizontal wind. Returns (frames, total_reward)."""
    env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    state, _ = env.reset(seed=seed)
    done, steps, total = False, 0, 0.0

    while not done and steps < max_steps:
        frames.append(env.render())
        if wind != 0.0:
            env.unwrapped.data.xfrc_applied[1, 0] = wind  # force on cart, x-axis
        with torch.no_grad():
            action = agent.sample_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
        steps += 1

    env.close()
    return frames, total


def eval_with_wind(agent, env_name: str, wind: float,
                   n_episodes: int = 5, max_steps: int = 1000):
    """Evaluate over n_episodes. Returns (mean_reward, std_reward)."""
    env = gym.make(env_name)
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        done, steps, total = False, 0, 0.0
        while not done and steps < max_steps:
            if wind != 0.0:
                env.unwrapped.data.xfrc_applied[1, 0] = wind
            with torch.no_grad():
                action = agent.sample_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
            steps += 1
        rewards.append(total)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# frame helpers

def add_label(frame: np.ndarray, label: str, subtitle: str = "") -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font_main = ImageFont.truetype("arial.ttf", 18)
        font_sub  = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font_main = font_sub = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 30], fill=(0, 0, 0, 200))
    draw.text((8, 6), label, fill=(255, 255, 255), font=font_main)
    if subtitle:
        draw.rectangle([0, img.height - 22, img.width, img.height], fill=(0, 0, 0, 160))
        draw.text((8, img.height - 18), subtitle, fill=(200, 200, 200), font=font_sub)
    return np.array(img)


def pad_frames(frames_list: list) -> list:
    max_len = max(len(f) for f in frames_list)
    return [
        f + [f[-1]] * (max_len - len(f)) if len(f) < max_len else f
        for f in frames_list
    ]


def save_video(frames: list, path: str, fps: int = 30):
    imageio.mimsave(path, frames, fps=fps)
    print(f"  Saved: {path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


# 4-panel robustness video (2×2) 

def make_robustness_video(agent, env_name: str, agent_label: str,
                          output_filename: str, max_steps: int = 500):
    print(f"\n[{output_filename}] Recording {agent_label}...")
    assert len(WIND_FORCES) == 4, "Expected exactly 4 wind levels for 2x2 grid"

    all_frames = []
    for wind in WIND_FORCES:
        frames, reward = record_with_wind(agent, env_name, wind, max_steps=max_steps)
        label    = f"{agent_label} | Wind = {wind} N"
        subtitle = f"Episode reward: {reward:.0f}"
        frames   = [add_label(f, label, subtitle) for f in frames]
        all_frames.append(frames)
        print(f"  wind={wind}N → {len(frames)} frames, reward={reward:.0f}")

    all_frames = pad_frames(all_frames)

    combined = []
    for f0, f1, f2, f3 in zip(*all_frames):
        top = np.hstack([f0, f1])
        bot = np.hstack([f2, f3])
        combined.append(np.vstack([top, bot]))

    save_video(combined, os.path.join(VIDEOS_DIR, output_filename))


# robustness plot 

def save_fig(fig, filename: str):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_robustness(results: list, title: str, filename: str):
    """
    results: list of dicts {label, color, means, stds}
    The fill_between shows ±1 std over N_EVAL_EPS evaluation episodes
    (i.e. real variability at each wind level).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.array(WIND_FORCES)

    for r in results:
        means = np.array(r["means"])
        stds  = np.array(r["stds"])
        ax.plot(x, means, marker="o", label=r["label"],
                color=r["color"], linewidth=2.2)
        ax.fill_between(x, means - stds, means + stds,
                        alpha=0.2, color=r["color"])

    ax.set_xlabel("Wind Force (N)")
    ax.set_ylabel(f"Mean Episode Reward (n={N_EVAL_EPS})")
    ax.set_title(title)
    ax.set_xticks(WIND_FORCES)
    ax.legend()
    save_fig(fig, filename)


# Main 

if __name__ == "__main__":

    # single pendulum
    print("\n=== Single Pendulum Robustness (seed 3) ===")
    ENV_SP = "InvertedPendulum-v5"

    vanilla_agent  = load_vanilla( "checkpoints/vanilla_seed3_ep5000.pth",  ENV_SP)
    baseline_agent = load_baseline("checkpoints/baseline_seed3_ep5000.pth", ENV_SP)

    # 4-panel videos
    make_robustness_video(vanilla_agent,  ENV_SP, "Vanilla REINFORCE",    "robustness_vanilla.mp4")
    make_robustness_video(baseline_agent, ENV_SP, "REINFORCE + Baseline", "robustness_baseline.mp4")

    # stats -> plot
    sp_results = []
    for agent, label, color in [
        (vanilla_agent,  "Vanilla REINFORCE",    "#1F77B4"),
        (baseline_agent, "REINFORCE + Baseline", "#2CA02C"),
    ]:
        means, stds = [], []
        for wind in WIND_FORCES:
            m, s = eval_with_wind(agent, ENV_SP, wind, N_EVAL_EPS)
            means.append(m); stds.append(s)
            print(f"  {label:30s} | wind={wind}N: {m:.1f} ± {s:.1f}")
        sp_results.append({"label": label, "color": color, "means": means, "stds": stds})

    plot_robustness(sp_results,
                    "Robustness to Wind - Single Pendulum (seed 3)",
                    "09_robustness_single_pendulum.png")

    # double pendulum
    print("\n=== Double Pendulum Robustness (seed 3) ===")
    ENV_DP = "InvertedDoublePendulum-v5"

    dp_configs = [
        ("checkpoints/double_pendulum_seed3_ep10000.pth",
         "From Scratch (seed3)",          "#9467BD", "robustness_dp_scratch.mp4"),
        ("checkpoints/transfer_seed3.pth",
         "Transfer Learning (seed3)",     "#D62728", "robustness_dp_transfer.mp4"),
    ]

    dp_results = []
    for path, label, color, vid_name in dp_configs:
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue

        agent = load_baseline(path, ENV_DP)

        # 4-panel video
        make_robustness_video(agent, ENV_DP, label, vid_name, max_steps=1000)

        # statistics
        means, stds = [], []
        for wind in WIND_FORCES:
            m, s = eval_with_wind(agent, ENV_DP, wind, N_EVAL_EPS, max_steps=1000)
            means.append(m); stds.append(s)
            print(f"  {label:35s} | wind={wind}N: {m:.1f} ± {s:.1f}")
        dp_results.append({"label": label, "color": color, "means": means, "stds": stds})

    if dp_results:
        plot_robustness(dp_results,
                        "Robustness to Wind - Double Pendulum (seed 3)",
                        "10_robustness_double_pendulum.png")

    print("\nDone! Check videos/ and plots/ directories.")
