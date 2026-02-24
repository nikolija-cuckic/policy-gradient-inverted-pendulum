import gymnasium as gym
import torch
import numpy as np
import imageio

from src.agent_baseline import REINFORCEWithBaseline


def record_episode(agent, env_name="InvertedPendulum-v4", max_steps=500, fps=30):
    """
    Snima jednu epizodu agenta i vraća listu frejmova.
    
    Args:
        agent: Treniran (ili netreniran) agent
        env_name: Ime okruženja
        max_steps: Maksimalan broj koraka
        fps: Frame rate (potreban za pravilnu dužinu videa)
    
    Returns:
        Lista RGB frame-ova
    """
    env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    
    state, _ = env.reset(seed=42)
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        frame = env.render()
        frames.append(frame)
        
        action = agent.sample_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    
    env.close()
    return frames


def create_comparison_video():
    """
    Pravi video sa poređenjem: Netreniran agent (levo) vs. Treniran agent (desno).
    """
    env_name = "InvertedPendulum-v4"
    obs_dim = gym.make(env_name).observation_space.shape[0]
    act_dim = gym.make(env_name).action_space.shape[0]
    
    # 1. Netreniran agent (randomizovan)
    print("Snimam netrenirani model...")
    untrained_agent = REINFORCEWithBaseline(obs_dim, act_dim)
    untrained_frames = record_episode(untrained_agent, env_name, max_steps=200)
    
    # 2. Treniran agent (učitaj checkpoint)
    print("Snimam trenirani model...")
    trained_agent = REINFORCEWithBaseline(obs_dim, act_dim)
    
    try:
        checkpoint = torch.load("checkpoints/baseline_seed1_ep3000.pth")
        trained_agent.policy_net.load_state_dict(checkpoint['policy'])
        trained_agent.value_net.load_state_dict(checkpoint['value'])
        print("✓ Model učitan")
    except FileNotFoundError:
        print("❌ GREŠKA: Nije pronađen 'checkpoints/baseline_seed1_ep3000.pth'")
        print("   Prvo pokreni: python main_train_baseline.py")
        return
    
    trained_frames = record_episode(trained_agent, env_name, max_steps=200)
    
    # 3. Izjednači dužine (možda će jedan agent ranije pasti)
    min_len = 30
    untrained_frames = untrained_frames[:min_len]
    trained_frames = trained_frames[:min_len]
    
    # 4. Napravi side-by-side video
    print("Kreiranje videa...")
    combined_frames = []
    
    for untrained_frame, trained_frame in zip(untrained_frames, trained_frames):
        # Spoji slike horizontalno
        combined = np.hstack([untrained_frame, trained_frame])
        combined_frames.append(combined)
    
    # 5. Dodaj tekst preko frejmova (opciono, zahteva PIL/opencv)
    # Ovde možeš dodati "Before" i "After" natpise ako želiš
    
    # 6. Snimi kao MP4
    output_path = "logs/comparison_video.mp4"
    imageio.mimsave(output_path, combined_frames, fps=30)
    
    print(f"✓ Video sačuvan: {output_path}")
    print(f"  Dužina: {len(combined_frames)} frejmova ({len(combined_frames)/30:.1f} sekundi)")


def create_single_video(model_path, output_name="trained_agent.mp4"):
    """
    Pravi video samo jednog agenta (jednostavnija verzija).
    
    Args:
        model_path: Putanja do .pth modela
        output_name: Ime output fajla
    """
    env_name = "InvertedPendulum-v4"
    obs_dim = gym.make(env_name).observation_space.shape[0]
    act_dim = gym.make(env_name).action_space.shape[0]
    
    agent = REINFORCEWithBaseline(obs_dim, act_dim)
    
    try:
        checkpoint = torch.load(model_path)
        agent.policy_net.load_state_dict(checkpoint['policy'])
        agent.value_net.load_state_dict(checkpoint['value'])
    except FileNotFoundError:
        print(f"❌ Model nije pronađen: {model_path}")
        return
    
    print(f"Snimam video iz modela: {model_path}")
    frames = record_episode(agent, env_name, max_steps=500)
    
    output_path = f"logs/{output_name}"
    imageio.mimsave(output_path, frames, fps=30)
    print(f"✓ Video sačuvan: {output_path}")


if __name__ == "__main__":
    # Instalacija (jednom):
    # pip install imageio[ffmpeg]
    
    # Opcija 1: Side-by-side poređenje
    create_comparison_video()
    
    # Opcija 2: Samo treniran agent
    create_single_video("checkpoints/baseline_seed1_ep1000.pth", "1000_agent.mp4")
    create_single_video("checkpoints/baseline_seed1_ep2000.pth", "2000_agent.mp4")
    create_single_video("checkpoints/baseline_seed1_ep3000.pth", "3000_agent.mp4")
