import gymnasium as gym
import numpy as np
from tqdm import tqdm
from src.agent_baseline import REINFORCEWithBaseline
from src.utils import set_seed, save_results_to_csv
import os
# Ovi parametri teraju PyTorch i NumPy da koriste sva jezgra
os.environ["OMP_NUM_THREADS"] = "4"  # Stavi broj fizičkih jezgara tvog Ryzena (verovatno 8)
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["IN_PARALLEL"] = "true"


def train(seed=1):
    env = gym.make("InvertedPendulum-v5")
    set_seed(seed)
    
    agent = REINFORCEWithBaseline(env.observation_space.shape[0], env.action_space.shape[0])
    rewards_history = []
    
    for i in tqdm(range(5000), desc=f"Baseline Seed {seed}"):
        state, _ = env.reset(seed=seed+i)
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
        
        if (i+1) % 1000 == 0:
            agent.save(f"checkpoints/baseline_seed{seed}_ep{i+1}.pth")
            
    save_results_to_csv(rewards_history, f"logs/baseline_seed{seed}.csv")

if __name__ == "__main__":
    for s in [3]:
        train(s)
