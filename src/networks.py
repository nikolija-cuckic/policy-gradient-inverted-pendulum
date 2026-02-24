import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """Outputs mean and std of a Gaussian distribution over actions."""
    def __init__(self, obs_space_dims, action_space_dims):
        super().__init__()
        hidden_space1 = 16
        hidden_space2 = 32

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )
        self.policy_mean_net   = nn.Linear(hidden_space2, action_space_dims)
        self.policy_stddev_net = nn.Linear(hidden_space2, action_space_dims)

    def forward(self, x):
        shared_features = self.shared_net(x.float())
        action_means    = self.policy_mean_net(shared_features)
        # Softplus ensures std is always positive
        action_stddevs  = torch.log(1 + torch.exp(self.policy_stddev_net(shared_features)))
        return action_means, action_stddevs


class ValueNetwork(nn.Module):
    """Estimates the state value V(s) used as the REINFORCE baseline."""
    def __init__(self, obs_space_dims):
        super().__init__()
        hidden_space1 = 32
        hidden_space2 = 32

        self.value_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2, 1)
        )

    def forward(self, x):
        return self.value_net(x.float())
