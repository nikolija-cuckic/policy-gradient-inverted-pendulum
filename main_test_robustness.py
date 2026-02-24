import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.agent_baseline import REINFORCEWithBaseline
from src.agent_vanilla import VanillaREINFORCE

def test_with_wind(agent_type, model_path, wind_force):
    env = gym.make("InvertedPendulum-v5")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    if agent_type == "vanilla":
        agent = VanillaREINFORCE(obs_dim, act_dim)
        # Vanilla čuva samo state_dict mreže
        agent.policy_net.load_state_dict(torch.load(model_path))
    else:
        agent = REINFORCEWithBaseline(obs_dim, act_dim)
        # Baseline čuva ceo dict
        ckpt = torch.load(model_path)
        agent.policy_net.load_state_dict(ckpt['policy'])

    rewards = []
    for _ in range(50): # Testiramo 50 epizoda
        state, _ = env.reset()
        done = False
        ep_reward = 0
        step_cnt = 0
        
        while not done:
            action = agent.sample_action(state)
            
            # --- VETAR ---
            if step_cnt % 10 == 0:
                # MuJoCo hack: qfrc_applied[1] je sila na štap
                env.unwrapped.data.qfrc_applied[1] = np.random.uniform(-wind_force, wind_force)
            
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step_cnt += 1
            
        rewards.append(ep_reward)
    return np.mean(rewards)

if __name__ == "__main__":
    winds = [0, 1, 5, 10]
    vanilla_res = []
    baseline_res = []
    
    # Učitavamo poslednje modele prvog seed-a
    # Putanje moraju postojati (pokreni trening prvo!)
    try:
        for w in winds:
            v = test_with_wind("vanilla", "checkpoints/vanilla_seed1_ep3000.pth", w)
            b = test_with_wind("baseline", "checkpoints/baseline_seed1_ep3000.pth", w)
            vanilla_res.append(v)
            baseline_res.append(b)
            print(f"Wind {w}: Vanilla={v:.1f}, Baseline={b:.1f}")
            
        # Crtanje grafika
        plt.plot(winds, vanilla_res, label='Vanilla REINFORCE', marker='o')
        plt.plot(winds, baseline_res, label='REINFORCE + Baseline', marker='o')
        plt.xlabel('Jačina vetra')
        plt.ylabel('Prosečna nagrada')
        plt.title('Test Robustnosti')
        plt.legend()
        plt.savefig('logs/robustness_chart.png')
        print("Grafik sačuvan u logs/robustness_chart.png")
    except FileNotFoundError:
        print("Greška: Nisu pronađeni modeli. Prvo pokreni main_train_vanilla.py i main_train_baseline.py!")
