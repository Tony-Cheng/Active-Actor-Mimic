import torch
from ..utils import to_policy


class EntropySelector:
    def __init__():
        raise NotImplementedError

    def select_samples(config, samples):
        raise NotImplementedError


def entropy(config, samples):
    states, actions, rewards, next_states, dones = samples
    num_states = states.size(0)
    batch_size = config.batch_size
    device = config.device
    entropy = torch.zeros((num_states), dtype=torch.float)
    for i in range(0, num_states, batch_size):
        batch_len = min(batch_size, num_states - i)
        batch_states = states[i: i + batch_len, :, :].to(device)
        with torch.no_grad():
            q_values = config.policy_net.generate_q_values(batch_states)
            policy = to_policy(config, q_values)
            num_q_values = policy.size(2)
            policy = torch.sum(policy, dim=2) / num_q_values
        current_entropy = - torch.sum(policy * torch.log(policy + 1e-8), 1)
        entropy[i: i + batch_len] = current_entropy.to('cpu')
    return entropy
