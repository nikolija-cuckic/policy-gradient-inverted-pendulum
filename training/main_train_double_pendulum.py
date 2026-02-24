import gymnasium as gym
import numpy as np
import os
import torch
from tqdm import tqdm
from src.agent_baseline import REINFORCEWithBaseline
from src.utils import set_seed, save_results_to_csv

# Use half the available CPU cores for PyTorch and MKL thread pools
num_cores = os.cpu_count() // 2
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
torch.set_num_threads(num_cores)

def train_double_pendulum(seed=1, total_episodes=5000):
    env_name = "InvertedDoublePendulum-v5"

    print(f"Training double pendulum (seed {seed}, {total_episodes} episodes)")
    env = gym.make(env_name)
    set_seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = REINFORCEWithBaseline(obs_dim, act_dim)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    rewards_history = []

    for episode in tqdm(range(total_episodes), desc=f"Double Pendulum Seed {seed}"):
        state, _ = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0

        while not done:
            action = agent.sample_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            ep_reward += reward
            done = terminated or truncated

        agent.update()
        rewards_history.append(ep_reward)

        if (episode + 1) % 1000 == 0:
            agent.save(f"checkpoints/double_pendulum_seed{seed}_ep{episode+1}.pth")

    # Save final model and reward log
    agent.save(f"checkpoints/double_pendulum_seed{seed}.pth")
    save_results_to_csv(rewards_history, f"logs/double_pendulum_seed{seed}.csv")
    print(f"Training complete. Results in logs/double_pendulum_seed{seed}.csv")

if __name__ == "__main__":
    train_double_pendulum(seed=3, total_episodes=10000)
