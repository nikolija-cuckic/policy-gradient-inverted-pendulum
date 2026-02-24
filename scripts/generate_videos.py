"""
generate_videos.py
------------------
Generates all MP4 videos for the RL exam project:
  - baseline_phases.mp4         : baseline agent at ep1000, ep3000, ep5000 (3-panel)
  - vanilla_phases.mp4          : vanilla agent at ep1000, ep3000, ep5000 (3-panel)
  - vanilla_vs_baseline.mp4     : vanilla vs baseline final models (2-panel)
  - double_pendulum.mp4         : double pendulum trained agent (single panel)
  - transfer_vs_scratch.mp4     : transfer learning vs scratch on double pendulum (2-panel)

Requirements:
  pip install imageio[ffmpeg] pillow
"""

import os
import numpy as np
import torch
import gymnasium as gym
import imageio
from PIL import Image, ImageDraw, ImageFont

from src.agent_vanilla import VanillaREINFORCE
from src.agent_baseline import REINFORCEWithBaseline

OUTPUT_DIR = "videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vanilla(checkpoint_path: str, env_name: str = "InvertedPendulum-v5") -> VanillaREINFORCE:
    env = gym.make(env_name)
    agent = VanillaREINFORCE(env.observation_space.shape[0], env.action_space.shape[0])
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    env.close()
    return agent


def load_baseline(checkpoint_path: str, env_name: str = "InvertedPendulum-v5") -> REINFORCEWithBaseline:
    env = gym.make(env_name)
    agent = REINFORCEWithBaseline(env.observation_space.shape[0], env.action_space.shape[0])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    agent.policy_net.load_state_dict(ckpt["policy"])
    agent.value_net.load_state_dict(ckpt["value"])
    env.close()
    return agent


def record_episode(agent, env_name: str, max_steps: int = 500, seed: int = 42) -> list:
    """Run one episode and collect RGB frames. Agent must NOT be in training mode."""
    env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    state, _ = env.reset(seed=seed)
    done = False
    steps = 0
    while not done and steps < max_steps:
        frames.append(env.render())
        with torch.no_grad():
            action = agent.sample_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    env.close()
    return frames


def add_label(frame: np.ndarray, label: str, subtitle: str = "") -> np.ndarray:
    """Burn a text label onto a frame using PIL."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font_main = ImageFont.truetype("arial.ttf", 18)
        font_sub  = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font_main = ImageFont.load_default()
        font_sub  = font_main

    # Dark background strip at top
    draw.rectangle([0, 0, img.width, 30], fill=(0, 0, 0, 200))
    draw.text((8, 6), label, fill=(255, 255, 255), font=font_main)
    if subtitle:
        draw.rectangle([0, img.height - 22, img.width, img.height], fill=(0, 0, 0, 160))
        draw.text((8, img.height - 18), subtitle, fill=(200, 200, 200), font=font_sub)
    return np.array(img)


def pad_frames(frames_list: list) -> list:
    """Pad shorter episode sequences by repeating the last frame."""
    max_len = max(len(f) for f in frames_list)
    padded = []
    for frames in frames_list:
        if len(frames) < max_len:
            frames = frames + [frames[-1]] * (max_len - len(frames))
        padded.append(frames)
    return padded


def save_video(frames: list, path: str, fps: int = 30):
    imageio.mimsave(path, frames, fps=fps)
    print(f"  Saved: {path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


# ---------------------------------------------------------------------------
# Video 1 & 2: Training-phase comparisons (3-panel, single environment)
# ---------------------------------------------------------------------------

def make_phase_video(
    load_fn,
    checkpoints: list,
    labels: list,
    env_name: str,
    output_filename: str,
    max_steps: int = 500,
):
    """
    Creates a side-by-side (N-panel) video showing different training phases.

    Args:
        load_fn: function(path, env_name) -> agent
        checkpoints: list of .pth paths
        labels: list of label strings (same length as checkpoints)
        env_name: gymnasium env name
        output_filename: output .mp4 filename (inside OUTPUT_DIR)
        max_steps: max steps per recorded episode
    """
    print(f"\n[{output_filename}] Recording {len(checkpoints)} phases...")

    all_frames = []
    for path, label in zip(checkpoints, labels):
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            # Substitute blank frames so the video still renders
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            all_frames.append([dummy] * 60)
            continue
        agent = load_fn(path, env_name)
        frames = record_episode(agent, env_name, max_steps=max_steps)
        frames = [add_label(f, label) for f in frames]
        all_frames.append(frames)
        print(f"  {label}: {len(frames)} frames")

    all_frames = pad_frames(all_frames)

    combined = []
    for step_frames in zip(*all_frames):
        row = np.hstack(list(step_frames))
        combined.append(row)

    save_video(combined, os.path.join(OUTPUT_DIR, output_filename))


# ---------------------------------------------------------------------------
# Video 3: Vanilla vs Baseline (2-panel)
# ---------------------------------------------------------------------------

def make_vanilla_vs_baseline_video():
    print("\n[vanilla_vs_baseline.mp4] Recording...")
    env_name = "InvertedPendulum-v5"

    vanilla_path  = "checkpoints/vanilla_seed1_ep5000.pth"
    baseline_path = "checkpoints/baseline_seed1_ep5000.pth"

    agents_info = [
        (vanilla_path,  load_vanilla,  "Vanilla REINFORCE (ep5000)"),
        (baseline_path, load_baseline, "REINFORCE + Baseline (ep5000)"),
    ]

    all_frames = []
    for path, load_fn, label in agents_info:
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            all_frames.append([np.zeros((480, 640, 3), dtype=np.uint8)] * 60)
            continue
        agent = load_fn(path, env_name)
        frames = record_episode(agent, env_name, max_steps=500)
        frames = [add_label(f, label) for f in frames]
        all_frames.append(frames)
        print(f"  {label}: {len(frames)} frames")

    all_frames = pad_frames(all_frames)
    combined = [np.hstack(list(sf)) for sf in zip(*all_frames)]
    save_video(combined, os.path.join(OUTPUT_DIR, "vanilla_vs_baseline.mp4"))


# ---------------------------------------------------------------------------
# Video 4: Double Pendulum (single panel)
# ---------------------------------------------------------------------------

def make_double_pendulum_video():
    print("\n[double_pendulum.mp4] Recording...")
    env_name = "InvertedDoublePendulum-v5"
    path = "checkpoints/double_pendulum_seed3_ep10000.pth"

    if not os.path.exists(path):
        path = "checkpoints/double_pendulum_seed3.pth"

    if not os.path.exists(path):
        print(f"  SKIP: no double pendulum checkpoint found.")
        return

    env = gym.make(env_name)
    agent = REINFORCEWithBaseline(env.observation_space.shape[0], env.action_space.shape[0])
    ckpt = torch.load(path, map_location="cpu")
    agent.policy_net.load_state_dict(ckpt["policy"])
    agent.value_net.load_state_dict(ckpt["value"])
    env.close()

    frames = record_episode(agent, env_name, max_steps=1000)
    frames = [add_label(f, "Double Pendulum – Trained from Scratch (ep10000)", "InvertedDoublePendulum-v5") for f in frames]
    save_video(frames, os.path.join(OUTPUT_DIR, "double_pendulum.mp4"))


# ---------------------------------------------------------------------------
# Video 5: Transfer Learning vs Scratch (2-panel, double pendulum)
# ---------------------------------------------------------------------------

def make_transfer_vs_scratch_video():
    print("\n[transfer_vs_scratch.mp4] Recording...")
    env_name = "InvertedDoublePendulum-v5"

    transfer_path = "checkpoints/transfer_seed3.pth"
    scratch_path  = "checkpoints/double_pendulum_seed3_ep10000.pth"

    labels = ["Transfer Learning – seed3 (ep10000)", "From Scratch – seed3 (ep10000)"]
    paths  = [transfer_path, scratch_path]

    all_frames = []
    for path, label in zip(paths, labels):
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            all_frames.append([np.zeros((480, 640, 3), dtype=np.uint8)] * 60)
            continue
        env = gym.make(env_name)
        agent = REINFORCEWithBaseline(env.observation_space.shape[0], env.action_space.shape[0])
        ckpt = torch.load(path, map_location="cpu")
        agent.policy_net.load_state_dict(ckpt["policy"])
        agent.value_net.load_state_dict(ckpt["value"])
        env.close()

        frames = record_episode(agent, env_name, max_steps=1000)
        frames = [add_label(f, label, "InvertedDoublePendulum-v5") for f in frames]
        all_frames.append(frames)
        print(f"  {label}: {len(frames)} frames")

    all_frames = pad_frames(all_frames)
    combined = [np.hstack(list(sf)) for sf in zip(*all_frames)]
    save_video(combined, os.path.join(OUTPUT_DIR, "transfer_vs_scratch.mp4"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Video 1: Baseline training phases ---
    make_phase_video(
        load_fn=load_baseline,
        checkpoints=[
            "checkpoints/baseline_seed1_ep1000.pth",
            "checkpoints/baseline_seed1_ep3000.pth",
            "checkpoints/baseline_seed1_ep5000.pth",
        ],
        labels=[
            "Baseline – ep1000 (early)",
            "Baseline – ep3000 (mid)",
            "Baseline – ep5000 (final)",
        ],
        env_name="InvertedPendulum-v5",
        output_filename="baseline_phases.mp4",
    )

    # --- Video 2: Vanilla training phases ---
    make_phase_video(
        load_fn=load_vanilla,
        checkpoints=[
            "checkpoints/vanilla_seed1_ep1000.pth",
            "checkpoints/vanilla_seed1_ep3000.pth",
            "checkpoints/vanilla_seed1_ep5000.pth",
        ],
        labels=[
            "Vanilla – ep1000 (early)",
            "Vanilla – ep3000 (mid)",
            "Vanilla – ep5000 (final)",
        ],
        env_name="InvertedPendulum-v5",
        output_filename="vanilla_phases.mp4",
    )

    # --- Video 3: Vanilla vs Baseline ---
    make_vanilla_vs_baseline_video()

    # --- Video 4: Double Pendulum ---
    make_double_pendulum_video()

    # --- Video 5: Transfer vs Scratch ---
    make_transfer_vs_scratch_video()

    print("\nDone! All videos saved to:", OUTPUT_DIR)
