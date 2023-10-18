import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.action_std_init = 1.5
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), self.action_std_init * self.action_std_init)

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, action_dim),
            nn.Sigmoid()
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        means = self.actor(state)*2-1
        value = self.critic(state)
        return means, value
    
if __name__ == '__main__':
    ppo = ActorCritic(3,2).to("cuda:0")
    # print(ppo)
    state = np.array([[1,2,3],[4,5,6]])
    state = torch.from_numpy(state).float().to("cuda:0")
    with torch.no_grad():
        means, value = ppo(state)
        cov_mat = torch.diag(ppo.action_var.to('cuda:0')).unsqueeze(dim=0)
        dist = MultivariateNormal(means, cov_mat)
        print(dist.sample())



