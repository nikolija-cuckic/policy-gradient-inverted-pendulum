import random
import numpy as np
import torch
import csv
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_results_to_csv(rewards, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward'])
        for i, r in enumerate(rewards):
            writer.writerow([i, r])
    print(f"Rezultati sačuvani u: {filepath}")
