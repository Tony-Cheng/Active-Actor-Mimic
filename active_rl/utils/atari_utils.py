import torch
import random
from ..environments import wrap_deepmind, make_atari
from collections import deque


class ActionSelector(object):
    def __init__(self, policy_net, INITIAL_EPSILON, FINAL_EPSILON, EPS_DECAY,
                 n_actions, EVAL_EPS=0.05, device='cuda'):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._EVAL_EPS = EVAL_EPS
        self._n_actions = n_actions
        self._device = device

    def select_action(self, state, training=True):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps or (not training and sample > self._EVAL_EPS):
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


def evaluate(policy_net, env_raw, action_selector, num_episode=5):
    env = wrap_deepmind(env_raw)
    sa = action_selector
    e_rewards = []
    q = deque(maxlen=5)
    for _ in range(num_episode):
        env.reset()
        img, _, _, _ = env.step(1)
        e_reward = 0
        for _ in range(10):  # no-op
            n_frame, _, done, _ = env.step(0)
            n_frame = fp(n_frame)
            q.append(n_frame)

        while not done:
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state, training=False)
            n_frame, reward, done, _ = env.step(action)
            n_frame = fp(n_frame)
            q.append(n_frame)

            e_reward += reward
        e_rewards.append(e_reward)

    return float(sum(e_rewards))/float(num_episode)


def make_raw_env(env_name):
    return make_atari('{}NoFrameskip-v4'.format(env_name))


def env_shape(env_raw):
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=False,
                        clip_rewards=True)
    c, h, w = fp(env.reset()).shape
    n_actions = env.action_space.n
    return c, h, w, n_actions
