import gymnasium as gym
import numpy as np
from tqdm import tqdm
from src.agent_vanilla import VanillaREINFORCE
from src.utils import set_seed, save_results_to_csv

def train(seed=1):
    env = gym.make("InvertedPendulum-v5")
    set_seed(seed)

    agent = VanillaREINFORCE(env.observation_space.shape[0], env.action_space.shape[0])
    rewards_history = []

    for i in tqdm(range(5000), desc=f"Vanilla Seed {seed}"):
        state, _ = env.reset(seed=seed + i)
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

        if (i + 1) % 1000 == 0:
            agent.save(f"checkpoints/vanilla_seed{seed}_ep{i+1}.pth")

    save_results_to_csv(rewards_history, f"logs/vanilla_seed{seed}.csv")

if __name__ == "__main__":
    for s in [1, 2, 3]:
        train(s)
