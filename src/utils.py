import random
import numpy as np
import torch
import csv
import os

def set_seed(seed):
    """Set random seed for reproducibility across random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_results_to_csv(rewards, filepath):
    """Write a list of per-episode rewards to a CSV file with header (episode, reward)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward'])
        for i, r in enumerate(rewards):
            writer.writerow([i, r])
    print(f"Results saved to: {filepath}")
