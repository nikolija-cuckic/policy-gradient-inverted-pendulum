import torch
from torch.distributions.normal import Normal
from src.networks import PolicyNetwork

class VanillaREINFORCE:
    def __init__(self, obs_dims, action_dims, lr=1e-4, gamma=0.99):
        self.policy_net = PolicyNetwork(obs_dims, action_dims)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def sample_action(self, state):
        state = torch.tensor(state)
        means, stds = self.policy_net(state)
        dist = Normal(means + 1e-6, stds + 1e-6)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.detach().numpy()

    def update(self):
        """Ažuriranje po formuli: ∇J ≈ Σ G_t * ∇log π(a|s)"""
        R = 0
        returns = []
        # Računamo G_t unazad
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        log_probs = torch.stack(self.log_probs).squeeze() # ako je dimenzija problem

        # GUBITAK: minus log_prob * return
        loss = -torch.sum(log_probs * returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
