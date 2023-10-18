import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
# from model import ActorCritic
from agents.PPO.model import ActorCritic
class RolloutBuffer:
    def __init__(self) -> None:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
class PPO:
    def __init__(self, input_dim,action_dim,cfg):
        self.policy = ActorCritic(input_dim, action_dim).to(cfg.device)
        self.old_policy = ActorCritic(input_dim, action_dim).to(cfg.device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.eps_clip = cfg.eps_clip
        self.buffer = RolloutBuffer()
        self.device = cfg.device
        self.K_epochs = cfg.K_epochs
        self.train = cfg.train

        self.MseLoss = nn.MSELoss()
   
    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            means,state_values = self.old_policy(state)
        if self.train:
            cov_mat = torch.diag(self.old_policy.action_var.to(self.device)).unsqueeze(dim=0)
            dist = MultivariateNormal(means, cov_mat)
            a = dist.sample()
            a_logprob = dist.log_prob(a)
            # import pdb;pdb.set_trace()
            self.buffer.actions.append(a)
            self.buffer.states.append(state)
            self.buffer.logprobs.append(a_logprob)
            self.buffer.state_values.append(state_values)
        else:
            a = means

        return a
    

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if True in is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()
        mean = torch.mean(advantages)
        std = torch.std(advantages)
        # 减去均值并除以标准差
        advantages = (advantages - mean) / std

        for _ in range(self.K_epochs):
            # import ipdb;ipdb.set_trace()
            means,state_values = self.policy(old_states)
            cov_mat = torch.diag(self.old_policy.action_var.to(self.device)).unsqueeze(dim=0)
            dist = MultivariateNormal(means, cov_mat)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self,path,i_ep):
        torch.save(self.old_policy.state_dict(), path + "PPO_model_"+str(i_ep))
        torch.save(self.optimizer.state_dict(), path + "PPO_optimizer_"+str(i_ep))

    def load(self, path,i_ep):
        self.old_policy.load_state_dict(torch.load(path+ "PPO_model_"+str(i_ep)))
        self.optimizer.load_state_dict(torch.load(path+ "PPO_optimizer_"+str(i_ep)))

class Config():
    def __init__(self):
        self.input_dim = 3
        self.action_dim = 6
        self.lr = 0.0001
        self.gamma = 0.9
        self.eps_clip = 0.1
        self.device = "cuda:0"
        self.K_epochs = 80
        self.train = True

if __name__ == '__main__':
    
    agent = PPO(3,2,cfg=Config())
    states = np.array([[1,2,3],[4,5,6],[7,8,9]])
    agent.buffer.rewards = [1,2,3]
    agent.buffer.is_terminals = [0,0,0]
    for state in states:
        agent.select_action(state)
    agent.update()
    # import ipdb;ipdb.set_trace()
    # print(agent.buffer.logprobs)

    

