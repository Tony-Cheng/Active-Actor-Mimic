from ..utils import BaseConfig
from .atari_wrappers import make_atari, wrap_deepmind
from .atari_utils import fp
from collections import deque
import torch


class EnvInterface:
    def get_state(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass


class DiscreteAtariEnv(EnvInterface):
    def __init__(self, config: BaseConfig, eval=False):
        self.env_raw = make_atari('{}NoFrameskip-v4'.format(config.env_name))
        if eval:
            self.env = wrap_deepmind()
        else:
            self.env = wrap_deepmind(self.env_raw, frame_stack=False,
                                     episode_life=False, clip_rewards=True)
        n_channels, height, width = fp(self.env.reset())
        self.n_channels = n_channels
        self.height = height
        self.width = width
        self.n_actions = self.env.action_space.n_actions
        self.frame_queue = deque(maxlen=5)

    def get_n_actions(self):
        return self.n_actions

    def get_state(self):
        return torch.cat(list(self.frame_queue))[1:].unsqueeze(0)

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.frame_queue.append(frame)
        self.next_state = self.get_state()
        return self.next_state, reward, done, info

    def reset(self):
        self.reset()
        for _ in range(10):
            frame, _, _, _ = self.env.step(0)
            frame = fp(frame)
            self.frame_queue.append(frame)
        return self.get_state()
