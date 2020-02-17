import numpy as np
import torch
import random
import math
from environments.atari_wrappers import wrap_deepmind
from collections import deque
import cv2


def get_state(obs):
    """
    Convert an open AI gym state to numpy state.
    """
    obs = _preprocess(obs)
    state = np.array(obs)
    # state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def select_action(policy_net, state, eps_threshold, n_actions=4, device='cuda'):
    """
    Select the next action.
    """
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def eps_decay(steps_done, EPS_START=1, EPS_END=0.01, EPS_DECAY=1000000):
    """
    Compute the current epsilon threshold based on the number of steps done.
    """
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)


def _to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def _downsample(img):
    return img[::2, ::2]


def _preprocess(img):
    return _to_grayscale(_downsample(img))


def transform_reward(reward):
    return np.sign(reward)


class ActionSelector(object):
    def __init__(self, INITIAL_EPSILON, FINAL_EPSILON, policy_net, EPS_DECAY, n_actions, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device

    def select_action(self, state, training=False):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps:
            with torch.no_grad():
                a = self._policy_net(state.to(self._device)).max(1)[
                    1].cpu().view(1, 1)
        else:
            a = torch.tensor([[random.randrange(self._n_actions)]],
                             device='cpu', dtype=torch.long)

        return a.numpy()[0, 0].item(), self._eps


def fp(n_frame):
    n_frame = torch.from_numpy(n_frame)
    h = n_frame.shape[-2]
    return n_frame.view(1, h, h)

def evaluate(step, policy_net, device, env, n_actions, eps=0.05, num_episode=5):
    env = wrap_deepmind(env)
    sa = ActionSelector(eps, eps, policy_net, 1, n_actions, device)
    e_rewards = []
    q = deque(maxlen=5)
    for _ in range(num_episode):
        env.reset()
        e_reward = 0
        for _ in range(10): # no-op
            n_frame, _, done, _ = env.step(0)
            n_frame = fp(n_frame)
            q.append(n_frame)

        while not done:
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state, True)
            n_frame, reward, done, _ = env.step(action)
            n_frame = fp(n_frame)
            q.append(n_frame)
            
            e_reward += reward
        e_rewards.append(e_reward)

    return float(sum(e_rewards))/float(num_episode)
