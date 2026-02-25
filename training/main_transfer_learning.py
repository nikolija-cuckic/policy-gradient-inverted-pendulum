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
    Copy weights from a smaller source model (4D input) to a larger target
    model (11D input). The first layer is copied column-wise up to the source
    width; remaining input columns stay randomly initialized. All subsequent
    layers are identical in size and are copied directly.
    """
    # policy network - first layer (input -> hidden)
    src_w  = source_agent.policy_net.shared_net[0].weight.data
    src_b  = source_agent.policy_net.shared_net[0].bias.data
    tgt_w  = target_agent.policy_net.shared_net[0].weight.data
    tgt_b  = target_agent.policy_net.shared_net[0].bias.data

    min_in = min(src_w.shape[1], tgt_w.shape[1])
    tgt_w[:, :min_in] = src_w[:, :min_in]
    tgt_b[:] = src_b[:]

    # Policy network - remaining layers (same size across models)
    try:
        target_agent.policy_net.shared_net[2].weight.data       = source_agent.policy_net.shared_net[2].weight.data
        target_agent.policy_net.shared_net[2].bias.data         = source_agent.policy_net.shared_net[2].bias.data
        target_agent.policy_net.policy_mean_net.weight.data     = source_agent.policy_net.policy_mean_net.weight.data
        target_agent.policy_net.policy_mean_net.bias.data       = source_agent.policy_net.policy_mean_net.bias.data
        target_agent.policy_net.policy_stddev_net.weight.data   = source_agent.policy_net.policy_stddev_net.weight.data
        target_agent.policy_net.policy_stddev_net.bias.data     = source_agent.policy_net.policy_stddev_net.bias.data
        print("Policy network transferred successfully.")
    except Exception as e:
        print(f"Partial policy transfer: {e}")

    # value network - first layer
    try:
        src_v_w = source_agent.value_net.value_net[0].weight.data
        tgt_v_w = target_agent.value_net.value_net[0].weight.data
        tgt_v_w[:, :min_in] = src_v_w[:, :min_in]

        target_agent.value_net.value_net[2].weight.data = source_agent.value_net.value_net[2].weight.data
        target_agent.value_net.value_net[4].weight.data = source_agent.value_net.value_net[4].weight.data
        print("Value network transferred successfully.")
    except Exception as e:
        print(f"Partial value transfer: {e}")


def train_transfer(seed=1, total_episodes=2000):
    # load the pre-trained source model (single pendulum)
    source_path = f"checkpoints/baseline_seed{seed}_ep5000.pth"
    if not os.path.exists(source_path):
        print(f"Source model not found: {source_path}")
        return

    dummy_env    = gym.make("InvertedPendulum-v5")
    source_agent = REINFORCEWithBaseline(
        dummy_env.observation_space.shape[0],
        dummy_env.action_space.shape[0]
    )
    checkpoint = torch.load(source_path)
    source_agent.policy_net.load_state_dict(checkpoint['policy'])
    source_agent.value_net.load_state_dict(checkpoint['value'])
    dummy_env.close()

    # create the target model (double pendulum)
    target_env   = gym.make("InvertedDoublePendulum-v5")
    target_agent = REINFORCEWithBaseline(
        target_env.observation_space.shape[0],
        target_env.action_space.shape[0]
    )

    # transfer weights from source to target
    transfer_weights(source_agent, target_agent)

    # fine-tune on the target environment
    print(f"Starting transfer learning fine-tuning (seed {seed}, {total_episodes} episodes)")
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
    target_agent.save(f"checkpoints/transfer_seed{seed}.pth")
    print(f"Transfer learning complete. Checkpoint: checkpoints/transfer_seed{seed}.pth")


if __name__ == "__main__":
    train_transfer(seed=3, total_episodes=10000)
