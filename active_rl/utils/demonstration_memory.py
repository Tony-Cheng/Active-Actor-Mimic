import pickle
import torch


class RolloutOfflineReplayMemory(object):
    def __init__(self, file_name, gamma, rollout_length):
        with open(file_name, "rb") as file:
            contents = pickle.load(file)
            self.m_states = contents['states']
            self.m_actions = contents['actions']
            self.m_rewards = contents['rewards']
            self.m_dones = contents['dones']
            self.m_rewards_rollout, self.m_offset_rollout = self.n_steps_rollout(
                self.m_rewards, self.m_dones, gamma, rollout_length)
        self.position = 0
        self.size = self.m_states.shape[0]

    def n_steps_rollout(self, rewards, dones, gamma, n):
        num_rewards = rewards.shape[0]
        rewards_rollout = torch.zeros((num_rewards, 1))
        offset_rollout = torch.zeros((num_rewards, 1))
        for i in range(num_rewards):
            j = 0
            rewards_rollout[i, 0] = rewards[i + j, 0]
            while j + 1 < n and i + j + 1 < num_rewards and not dones[i + j, 0]:
                j += 1
                rewards_rollout[i, 0] += (gamma ** j) * rewards[i + j, 0]
            offset_rollout[i, 0] = j
        return rewards_rollout, offset_rollout

    def sample(self, batch_size):
        i = torch.randint(0, high=self.size, size=(batch_size,))
        states = self.m_states[i, :4]
        next_states = self.m_states[i, 1:]
        actions = self.m_actions[i]
        rewards = self.m_rewards[i].float()
        dones = self.m_dones[i].float()
        rollout_rewards = self.m_rewards_rollout[i].float()
        rollout_offsets = self.m_offset_rollout[i]
        rollout_next_states = self.m_states[(
            i + rollout_offsets.squeeze()).long(), 1:]
        rollout_dones = self.m_dones[(
            i + rollout_offsets.squeeze()).long()].float()
        return states, actions, rewards, next_states, dones, rollout_rewards, rollout_offsets, rollout_next_states, rollout_dones

    def __len__(self):
        return self.size


class GenericRankedRolloutOfflineReplayMemory(object):
    def __init__(self, file_name, gamma, rollout_length, rank_func):
        with open(file_name, "rb") as file:
            contents = pickle.load(file)
            self.m_states = contents['states']
            self.m_actions = contents['actions']
            self.m_rewards = contents['rewards']
            self.m_dones = contents['dones']
            self.m_rewards_rollout, self.m_offset_rollout = self.n_steps_rollout(
                self.m_rewards, self.m_dones, gamma, rollout_length)
        self.rank_func = rank_func
        self.position = 0
        self.size = self.m_states.shape[0]

    def n_steps_rollout(self, rewards, dones, gamma, n):
        num_rewards = rewards.shape[0]
        rewards_rollout = torch.zeros((num_rewards, 1))
        offset_rollout = torch.zeros((num_rewards, 1))
        for i in range(num_rewards):
            j = 0
            rewards_rollout[i, 0] = rewards[i + j, 0]
            while j + 1 < n and i + j + 1 < num_rewards and not dones[i + j, 0]:
                j += 1
                rewards_rollout[i, 0] += (gamma ** j) * rewards[i + j, 0]
            offset_rollout[i, 0] = j
        return rewards_rollout, offset_rollout

    def sample(self, batch_size):
        _, i = torch.sort(self.rank_func(self.m_states[: self.size, :4]),
                          descending=True)
        i = i[: batch_size]
        states = self.m_states[i, :4]
        next_states = self.m_states[i, 1:]
        actions = self.m_actions[i]
        rewards = self.m_rewards[i].float()
        dones = self.m_dones[i].float()
        rollout_rewards = self.m_rewards_rollout[i].float()
        rollout_offsets = self.m_offset_rollout[i]
        rollout_next_states = self.m_states[(
            i + rollout_offsets.squeeze()).long(), 1:]
        rollout_dones = self.m_dones[(
            i + rollout_offsets.squeeze()).long()].float()
        return states, actions, rewards, next_states, dones, rollout_rewards, rollout_offsets, rollout_next_states, rollout_dones

    def __len__(self):
        return self.size


class GenericRankedDoubleStatesRolloutOfflineReplayMemory(object):
    def __init__(self, file_name, gamma, rollout_length, rank_func):
        with open(file_name, "rb") as file:
            contents = pickle.load(file)
            self.m_states = contents['states']
            self.m_actions = contents['actions']
            self.m_rewards = contents['rewards']
            self.m_dones = contents['dones']
            self.m_rewards_rollout, self.m_offset_rollout = self.n_steps_rollout(
                self.m_rewards, self.m_dones, gamma, rollout_length)
        self.rank_func = rank_func
        self.position = 0
        self.size = self.m_states.shape[0]

    def n_steps_rollout(self, rewards, dones, gamma, n):
        num_rewards = rewards.shape[0]
        rewards_rollout = torch.zeros((num_rewards, 1))
        offset_rollout = torch.zeros((num_rewards, 1))
        for i in range(num_rewards):
            j = 0
            rewards_rollout[i, 0] = rewards[i + j, 0]
            while j + 1 < n and i + j + 1 < num_rewards and not dones[i + j, 0]:
                j += 1
                rewards_rollout[i, 0] += (gamma ** j) * rewards[i + j, 0]
            offset_rollout[i, 0] = j
        return rewards_rollout, offset_rollout

    def sample(self, batch_size):
        current_states = self.m_states[: self.size, :4]
        current_rollout_offsets = self.m_offset_rollout.squeeze().long()
        current_rollout_states = self.m_states[torch.arange(
            self.size) + current_rollout_offsets, 1:]
        _, i = torch.sort(self.rank_func(
            current_states, current_rollout_states), descending=True)
        i = i[: batch_size]
        states = self.m_states[i, :4]
        next_states = self.m_states[i, 1:]
        actions = self.m_actions[i]
        rewards = self.m_rewards[i].float()
        dones = self.m_dones[i].float()
        rollout_rewards = self.m_rewards_rollout[i].float()
        rollout_offsets = self.m_offset_rollout[i]
        rollout_next_states = self.m_states[(
            i + rollout_offsets.squeeze()).long(), 1:]
        rollout_dones = self.m_dones[(
            i + rollout_offsets.squeeze()).long()].float()
        return states, actions, rewards, next_states, dones, rollout_rewards, rollout_offsets, rollout_next_states, rollout_dones

    def __len__(self):
        return self.size
