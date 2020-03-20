from .config import DiscreteActionConfig
import torch


class Replay:
    def __init__(self, config: DiscreteActionConfig):

        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        height = config.height
        width = config.width
        self.num_channels = config.memory_channel
        self.frame_channel = config.frame_channel
        self.m_states = torch.zeros(
            (self.memory_size, self.num_channels, height, width),
            dtype=torch.uint8)
        self.m_actions = torch.zeros((self.memory_size, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((self.memory_size, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((self.memory_size, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.memory_size
        self.size = max(self.size, self.position)

    def sample(self):
        idx = torch.randint(0, high=self.size, size=(self.batch_size,))
        states = self.m_states[idx, :self.num_channels - self.frame_channel]
        next_states = self.m_states[idx, self.frame_channel:]
        actions = self.m_actions[idx]
        rewards = self.m_rewards[idx].float()
        dones = self.m_dones[idx].float()
        return states, actions, rewards, next_states, dones
