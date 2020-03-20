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
            self.env = wrap_deepmind(self.env_raw)
        else:
            self.env = wrap_deepmind(self.env_raw, frame_stack=False,
                                     episode_life=False, clip_rewards=True)
        frame_channel, height, width = fp(self.env.reset()).shape
        self.frame_channel = frame_channel
        self.height = height
        self.width = width
        self.n_actions = self.env.action_space.n
        self.n_channel = 5
        self.frame_queue = deque(maxlen=self.n_channel)

    def get_shape(self):
        return self.frame_channel, self.height, self.width, self.n_actions,\
            self.n_channel

    def get_state(self):
        return torch.cat(list(self.frame_queue))[1:].unsqueeze(0)

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.frame_queue.append(fp(frame))
        self.next_state = self.get_state()
        return self.next_state, reward, done, info

    def get_all_states(self):
        return torch.cat(list(self.frame_queue)).unsqueeze(0)

    def reset(self):
        self.env.reset()
        for _ in range(10):
            frame, _, _, _ = self.env.step(0)
            self.frame_queue.append(fp(frame))
        return self.get_state()
