import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import DiscreteActionConfig, AMNConfig
from enum import Enum


class BaseAgent(nn.Module):
    def train(self):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError


class EnsembleActionPolicy(Enum):
    AVERAGE_POLICY = 0


class DDQNAgent(nn.Module):

    def __init__(self, config: DiscreteActionConfig):
        super(DDQNAgent, self).__init__()
        self.body = config.body
        self.policy_net = self.body(config)
        self.target_net = self.body(config)
        self.update_target()
        self.target_net.eval()
        self.device = config.device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.memory = config.memory
        self.gamma = config.gamma
        self.optimizer = config.optimizer(
            self.policy_net.parameters(), lr=config.lr, eps=1.5e-4)
        self.config = config

    def forward(self, state):
        return self.policy_net(state)

    def train(self):
        samples = self.memory.sample()
        states, actions, rewards, next_states, dones = samples

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions)

        # Compute the expected Q values
        expected_state_action_values = (
            next_q_values * self.gamma)*(1.-dones) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            q_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.detach()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net, f'models/{filename}')


class DQNAgent(DDQNAgent):
    def __init__(self, config: DiscreteActionConfig):
        super(DQNAgent, self).__init__(config)

    def train(self):
        samples = self.memory.sample()
        states, actions, rewards, next_states, dones = samples

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (
            next_q_values * self.gamma)*(1.-dones[:, 0]) + rewards[:, 0]

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            q_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.detach()


class DDQNEnsembleAgent(nn.Module):

    def __init__(self, config: DiscreteActionConfig):
        super(DDQNEnsembleAgent, self).__init__()
        self.body = config.body
        self.num_ensembles = config.num_ensembles
        self.policy_net = nn.ModuleList(
            [self.body(config) for _ in range(self.num_ensembles)])
        self.target_net = nn.ModuleList(
            [self.body(config) for _ in range(self.num_ensembles)])
        self.update_target()
        self.target_net.eval()
        self.device = config.device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.memory = config.memory
        self.gamma = config.gamma
        self.optimizer = config.optimizer(
            self.policy_net.parameters(), lr=config.lr, eps=1.5e-4)
        self.config = config
        self.action_selection_policy = config.action_selection_policy

    def forward(self, state):
        if self.action_selection_policy is None:
            policy = 0
            for i in range(self.num_ensembles):
                policy += self.policy_net(state)
            return policy / self.num_ensembles

    def train(self):
        samples = self.memory.sample()
        states, actions, rewards, next_states, dones = samples

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        total_loss = 0

        for i in range(self.num_ensembles):
            q_values = self.policy_net[i](states).gather(1, actions)
            next_actions = self.policy_net[i](
                next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net[i](
                next_states).gather(1, next_actions)

            # Compute the expected Q values
            expected_state_action_values = (
                next_q_values * self.gamma)*(1.-dones) + rewards

            # Compute Huber loss
            loss = F.smooth_l1_loss(
                q_values, expected_state_action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            total_loss += loss.detach()

        return total_loss / self.num_ensembles

    def update_target(self):
        for i in range(self.num_ensembles):
            self.target_net[i].load_state_dict(self.policy_net[i].state_dict())

    def save(self, filename):
        torch.save(self.policy_net, f'models/{filename}')


class DDQNAMNAgent(nn.Module):
    def __init__(self, config: AMNConfig):
        super(DDQNAMNAgent, self).__init__()
        self.body = config.body
        self.policy_net = self.body(config)
        self.target_net = torch.load(config.expert_net_name)
        self.target_net.eval()
        self.device = config.device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.memory = config.memory
        self.gamma = config.gamma
        self.optimizer = config.optimizer(
            self.policy_net.parameters(), lr=config.lr, eps=1.5e-4)
        self.config = config
        self.action_selection_policy = config.action_selection_policy

    def forward(self, state):
        return self.policy_net(state)

    def train(self):
        samples = self.memory.sample()
        states, actions, rewards, next_states, dones = samples
        states = states.to(self.device)

        AMN_q_value = self.policy_net(states)
        expert_q_value = self.target_net(states).detach()

        AMN_policy = to_policy(AMN_q_value)
        expert_policy = to_policy(expert_q_value)

        loss = -torch.sum(expert_policy * torch.log(AMN_policy + 1e-8))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()


class DDQNAMNEmsembleAgent(nn.Module):
    def __init__(self, config: AMNConfig):
        super(DDQNAMNAgent, self).__init__()
        self.body = config.body
        self.num_ensembles = config.num_ensembles
        self.policy_net = nn.ModuleList(
            [self.body(config) for _ in range(self.num_ensembles)])
        self.target_net = torch.load(config.expert_net_name)
        self.target_net.eval()
        self.device = config.device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.memory = config.memory
        self.gamma = config.gamma
        self.optimizer = config.optimizer(
            self.policy_net.parameters(), lr=config.lr, eps=1.5e-4)
        self.config = config

    def forward(self, state):
        if self.action_selection_policy is None:
            policy = 0
            for i in range(self.num_ensembles):
                policy += self.policy_net(state)
            return policy / self.num_ensembles

    def train(self):
        samples = self.memory.sample()
        states, actions, rewards, next_states, dones = samples
        states = states.to(self.device)

        total_loss = 0

        for i in range(self.num_ensembles):
            AMN_q_value = self.policy_net(states)
            expert_q_value = self.target_net(states).detach()

            AMN_policy = to_policy(AMN_q_value)
            expert_policy = to_policy(expert_q_value)

            loss = -torch.sum(expert_policy * torch.log(AMN_policy + 1e-8))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach()

        return total_loss


def to_policy(q_values, tau=0.1):
    return F.softmax(q_values / tau, dim=1)
