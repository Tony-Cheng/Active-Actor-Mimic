import torch
from ..environments.atari_wrappers import wrap_deepmind
from collections import deque
from .action_selection import MeanActionSelector
from .atari_utils import fp


def evaluate_action_mean(step, policy_net, device, env, n_actions, eps=0.05,
                         num_episode=5):
    env = wrap_deepmind(env)
    sa = MeanActionSelector(policy_net, eps, eps, 1, n_actions, device)
    e_rewards = []
    q = deque(maxlen=5)
    for _ in range(num_episode):
        env.reset()
        e_reward = 0
        for _ in range(10):  # no-op
            n_frame, _, done, _ = env.step(0)
            n_frame = fp(n_frame)
            q.append(n_frame)

        while not done:
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps, _ = sa.select_action(state, training=False)
            n_frame, reward, done, _ = env.step(action)
            n_frame = fp(n_frame)
            q.append(n_frame)

            e_reward += reward
        e_rewards.append(e_reward)

    return float(sum(e_rewards))/float(num_episode)
