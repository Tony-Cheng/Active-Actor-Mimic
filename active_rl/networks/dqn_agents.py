import torch
import torch.nn as nn
import torch.functional as F
from ..utils import DiscreteActionConfig


class BaseAgent(nn.Module):
    def train(self):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError


class DDQNAgent(nn.Module):

    def __init__(self, config: DiscreteActionConfig):
        super(DDQNAgent, self).__init__()
        self.body = config.body
        self.policy_net = self.body(config)
        self.target_net = self.body(config)
        self.update_target()
        self.device = config.device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.memory = config.memory
        self.gamma = config.gamma
        self.optimizer = config.optimizer(self.policy_net.state_dict())
        self.config = config

    def forward(self, state):
        return self.policy_net(state)

    def train(self):
        samples = self.memory.samples()
        states, actions, rewards, next_states, dones = samples

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        q_values = self.policy_netpolicy_net(states).gather(1, actions)
        next_actions = self.policy_net(next_states).max(1)[1].detach()
        next_q_values = self.target_net(
            next_states).gather(1, next_actions).detach()

        expected_state_action_values = (
            next_q_values * self.gamma) * (1.-dones[:, 0]) + rewards[:, 0]

        loss = F.smooth_l1_loss(q_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self, filename)
