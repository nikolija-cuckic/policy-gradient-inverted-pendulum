import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from src.agent_baseline import REINFORCEWithBaseline
from src.utils import set_seed, save_results_to_csv

num_cores = os.cpu_count() // 2
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
torch.set_num_threads(num_cores)

def transfer_weights(source_agent, target_agent):
    """
    Pametno kopiranje težina iz manjeg u veći model.
    Kopira težine za prvih 4 ulaza, ostale ostavlja random.
    """
    print("🔄 Započinjem transfer znanja...")
    
    # 1. Transfer POLICY mreže
    # Prvi sloj (input -> hidden)
    src_w = source_agent.policy_net.shared_net[0].weight.data
    src_b = source_agent.policy_net.shared_net[0].bias.data
    
    target_w = target_agent.policy_net.shared_net[0].weight.data
    target_b = target_agent.policy_net.shared_net[0].bias.data
    
    # Kopiraj težine (samo onoliko koliko staje)
    min_in = min(src_w.shape[1], target_w.shape[1])
    target_w[:, :min_in] = src_w[:, :min_in]
    target_b[:] = src_b[:] # Bias je isti (veličina hidden sloja je ista)
    
    # Ostali slojevi su iste veličine, pa možemo direktno kopirati
    # (Osim ako si menjala hidden_size u networks.py)
    try:
        target_agent.policy_net.shared_net[2].weight.data = source_agent.policy_net.shared_net[2].weight.data
        target_agent.policy_net.shared_net[2].bias.data = source_agent.policy_net.shared_net[2].bias.data
        
        target_agent.policy_net.policy_mean_net.weight.data = source_agent.policy_net.policy_mean_net.weight.data
        target_agent.policy_net.policy_mean_net.bias.data = source_agent.policy_net.policy_mean_net.bias.data
        
        target_agent.policy_net.policy_stddev_net.weight.data = source_agent.policy_net.policy_stddev_net.weight.data
        target_agent.policy_net.policy_stddev_net.bias.data = source_agent.policy_net.policy_stddev_net.bias.data
        
        print("✓ Policy mreža uspešno transferovana!")
    except Exception as e:
        print(f"⚠️ Delimičan transfer policy mreže: {e}")

    # 2. Transfer VALUE mreže (isto kao gore)
    try:
        # Prvi sloj
        src_v_w = source_agent.value_net.value_net[0].weight.data
        target_v_w = target_agent.value_net.value_net[0].weight.data
        target_v_w[:, :min_in] = src_v_w[:, :min_in]
        
        # Ostali slojevi
        target_agent.value_net.value_net[2].weight.data = source_agent.value_net.value_net[2].weight.data
        target_agent.value_net.value_net[4].weight.data = source_agent.value_net.value_net[4].weight.data
        print("✓ Value mreža uspešno transferovana!")
    except Exception as e:
        print(f"⚠️ Delimičan transfer value mreže: {e}")

def train_transfer(seed=1, total_episodes=2000):
    # 1. Učitaj SOURCE model (Single Pendulum)
    # Pretpostavljamo da imaš ovaj fajl
    source_path = f"checkpoints/baseline_seed{seed}_ep5000.pth"
    if not os.path.exists(source_path):
        print(f"❌ Nema izvornog modela: {source_path}")
        return

    # Kreiraj dummy okruženje samo da inicijalizujemo agenta
    dummy_env = gym.make("InvertedPendulum-v5")
    source_agent = REINFORCEWithBaseline(dummy_env.observation_space.shape[0], dummy_env.action_space.shape[0])
    checkpoint = torch.load(source_path)
    source_agent.policy_net.load_state_dict(checkpoint['policy'])
    source_agent.value_net.load_state_dict(checkpoint['value'])
    
    # 2. Kreiraj TARGET model (Double Pendulum)
    target_env = gym.make("InvertedDoublePendulum-v5")
    target_agent = REINFORCEWithBaseline(target_env.observation_space.shape[0], target_env.action_space.shape[0])
    
    # 3. Izvrši TRANSFER
    transfer_weights(source_agent, target_agent)
    
    # 4. Treniraj (Fine-tuning)
    print(f"--- Počinjem Transfer Learning (Seed {seed}) ---")
    rewards_history = []
    
    for episode in tqdm(range(total_episodes), desc="Transfer Learning"):
        state, _ = target_env.reset(seed=seed + episode)
        done = False
        ep_reward = 0
        
        while not done:
            action = target_agent.sample_action(state)
            state, reward, terminated, truncated, _ = target_env.step(action)
            target_agent.rewards.append(reward)
            ep_reward += reward
            done = terminated or truncated
            
        target_agent.update()
        rewards_history.append(ep_reward)
        
    save_results_to_csv(rewards_history, f"logs/transfer_learning_seed{seed}.csv")
    print(f"✓ Transfer Learning završen!")

if __name__ == "__main__":
    train_transfer(seed=1)
