import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.agent_baseline import REINFORCEWithBaseline

def capture_frames(model_path, env_name="InvertedPendulum-v4", num_frames=5):
    """Učitava model i vraća listu frejmova iz jedne epizode"""
    env = gym.make(env_name, render_mode="rgb_array")
    
    # Inicijalizacija agenta
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = REINFORCEWithBaseline(obs_dim, act_dim)
    
    # Učitaj model
    try:
        checkpoint = torch.load(model_path)
        agent.policy_net.load_state_dict(checkpoint['policy'])
    except FileNotFoundError:
        print(f"Model nije nađen: {model_path}")
        return None

    # Pusti epizodu
    frames = []
    state, _ = env.reset(seed=42) # Fiksni seed da uvek bude ista situacija
    done = False
    
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.sample_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if len(frames) > 200: # Prekini ako traje predugo
            break
            
    env.close()
    
    # Odaberi N frejmova ravnomerno raspoređenih
    indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    selected_frames = [frames[i] for i in indices]
    
    return selected_frames

def create_film_strip():
    # Definiši koje checkpointe želiš da prikažeš
    # Prilagodi imena fajlova onome što imaš u 'checkpoints/' folderu
    checkpoints = [
        ("Početak (Epizoda 0)", "checkpoints/baseline_seed1_ep1000.pth"), # Ili neki rani checkpoint
        ("Sredina (Epizoda 2000)", "checkpoints/baseline_seed1_ep2000.pth"),
        ("Kraj (Epizoda 3000)", "checkpoints/baseline_seed1_final.pth")
    ]
    
    num_rows = len(checkpoints)
    num_cols = 5 # Koliko sličica po redu
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    
    for row_idx, (label, model_path) in enumerate(checkpoints):
        print(f"Generišem slike za: {label}...")
        frames = capture_frames(model_path, num_frames=num_cols)
        
        if frames is None:
            continue
            
        for col_idx, frame in enumerate(frames):
            ax = axes[row_idx, col_idx]
            ax.imshow(frame)
            ax.axis('off')
            
            # Dodaj naslove
            if col_idx == 0:
                ax.set_title(label, loc='left', fontsize=14, fontweight='bold', x=-0.1, y=0.5)
            
            # Dodaj strelicu za vreme na dnu
            if row_idx == num_rows - 1 and col_idx == 2:
                ax.text(0.5, -0.2, "Vreme u epizodi ➝", transform=ax.transAxes, 
                       ha='center', fontsize=12)

    plt.suptitle("Evolucija REINFORCE Agenta: Od Padanja do Balansa", fontsize=16)
    output_path = "logs/film_strip.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Slika sačuvana: {output_path}")
    plt.show()

if __name__ == "__main__":
    create_film_strip()
