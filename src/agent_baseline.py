import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from src.networks import PolicyNetwork, ValueNetwork

class REINFORCEWithBaseline:
    def __init__(self, obs_dims, action_dims, lr=1e-4, gamma=0.99):
        self.policy_net = PolicyNetwork(obs_dims, action_dims)
        self.value_net  = ValueNetwork(obs_dims)

        self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.value_optim  = torch.optim.AdamW(self.value_net.parameters(),  lr=1e-3)

        self.gamma     = gamma
        self.log_probs = []
        self.rewards   = []
        self.states    = []

    def sample_action(self, state):
        state_t = torch.tensor(np.array([state])).float()
        means, stds = self.policy_net(state_t)
        dist = Normal(means + 1e-6, stds + 1e-6)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.states.append(state_t)

        return action.detach().numpy().flatten()

    def update(self):
        # REINFORCE with baseline: theta <- theta + alpha * (G_t - V(s)) * grad log pi(a|s)
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()

        states_tensor = torch.stack(self.states)
        values = self.value_net(states_tensor).squeeze()

        # update value network to minimize  error (V(s) -> G_t)
        value_loss = F.mse_loss(values, returns)
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        # update policy network using advantage = G_t - V(s)
        with torch.no_grad():
            values = self.value_net(states_tensor).squeeze()

        advantage   = returns - values
        log_probs   = torch.stack(self.log_probs).squeeze()
        policy_loss = -torch.sum(log_probs * advantage)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.log_probs = []
        self.rewards   = []
        self.states    = []

    def save(self, path):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'value':  self.value_net.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.value_net.load_state_dict(ckpt['value'])
