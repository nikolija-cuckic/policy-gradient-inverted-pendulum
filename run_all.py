import os
import time
from main_train_double_pendulum import train_double_pendulum
from main_transfer_learning import train_transfer

# OPTIMIZACIJA CPU (mora i ovde za svaki slučaj)
num_cores = os.cpu_count() // 2
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

if __name__ == "__main__":
    start_time = time.time()
    
    print("🚀 POČINJEM KOMPLETAN TRENING (Double Pendulum)...")
    print("="*60)
    
    # 1. Treniraj Double Pendulum od nule
    print("\n[1/2] Pokrećem trening od nule...")
    train_double_pendulum(seed=1, total_episodes=8000)
    
    # 2. Treniraj Transfer Learning
    print("\n[2/2] Pokrećem Transfer Learning...")
    # Pretpostavlja da imaš 'checkpoints/baseline_seed1_final.pth' od ranije!
    train_transfer(seed=1, total_episodes=3000)
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    
    print("="*60)
    print(f"✅ SVE ZAVRŠENO ZA {duration:.1f} MINUTA!")
    print("Sada možeš pokrenuti 'plot_transfer_comparison.py'")
