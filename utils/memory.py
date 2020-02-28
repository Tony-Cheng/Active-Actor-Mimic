from collections import namedtuple
import random
import torch
import cv2


class ReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, device):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, bs):
        if bs is None:
            bs = len(self)
        i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i, :4].to(self.device)
        bns = self.m_states[i, 1:].to(self.device)
        ba = self.m_actions[i].to(self.device)
        br = self.m_rewards[i].to(self.device).float()
        bd = self.m_dones[i].to(self.device).float()
        return bs, ba, br, bns, bd

    def __len__(self):
        return self.size


class RankedReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, rank_func, AMN_net, replacement=False, device='cuda'):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0
        self.rank_func = rank_func
        self.AMN_net = AMN_net
        self.replacement = replacement

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, percentage=0.1):
        _, i = torch.sort(self.rank_func(
            self.AMN_net, self.m_states[: self.size, :4], device=self.device), descending=True)
        i = i[: int(percentage * self.size)]
        i = i[torch.randperm(i.shape[0])]
        # i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i, :4]
        bns = self.m_states[i, 1:]
        ba = self.m_actions[i]
        br = self.m_rewards[i].float()
        bd = self.m_dones[i].float()
        if self.replacement:
            read_index = 0
            write_index = 0
            i_set = set(i.flatten())
            while read_index < self.size:
                if read_index in i_set:
                    read_index += 1
                else:
                    self.m_states[write_index] = self.m_states[read_index]
                    self.m_actions[write_index,
                                   0] = self.m_actions[read_index, 0]
                    self.m_rewards[write_index,
                                   0] = self.m_rewards[read_index, 0]
                    self.m_dones[write_index, 0] = self.m_dones[read_index, 0]
                    read_index += 1
                    write_index += 1
            self.size -= len(i_set)
            self.position -= len(i_set)
            self.position = self.position % self.size
        return bs, ba, br, bns, bd

    def __len__(self):
        return self.size


class _RankedReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, rank_func, AMN_net, device='cuda'):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0
        self.rank_func = rank_func
        self.AMN_net = AMN_net

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, percentage=0.1):
        _, i = torch.sort(self.rank_func(
            self.AMN_net, self.m_states[: self.size, :4], device=self.device), descending=True)
        i = i[: int(percentage * self.size)]
        i = i[torch.randperm(i.shape[0])]
        # i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i]
        ba = self.m_actions[i]
        br = self.m_rewards[i].float()
        bd = self.m_dones[i].float()
        return bs, ba, br, bd

    def __len__(self):
        return self.size


class LabeledReplayMemory():
    def __init__(self, capacity_not_labelled, capacity_labelled, state_shape,
                 n_actions, rank_func, AMN_net, device='cuda'):
        self.device = device
        self.labeled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.rank_buffer = _RankedReplayMemory(
            capacity_not_labelled, state_shape, n_actions, rank_func, AMN_net, device=device)

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.rank_buffer.push(state, action, reward, done)

    def label_sample(self, percentage=0.1):
        bs, ba, br, bd = self.rank_buffer.sample(percentage=percentage)
        for i in range(bs.shape[0]):
            self.labeled_buffer.push(bs[i], ba[i, 0], br[i, 0], bd[i, 0])
        return bs.shape[0]

    def sample(self, batch_szie=None):
        bs, ba, br, bns, bd = self.labeled_buffer.sample(bs=batch_szie)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.rank_buffer)
