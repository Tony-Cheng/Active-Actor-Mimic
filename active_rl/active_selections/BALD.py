from .entropy import entropy
from ..utils import to_policy
from ..utils.config import AMNConfig, BaseConfig
import torch


class BALDSelector:
    def __init__(self, config: AMNConfig):
        self.policy_net = config.policy_net
        self.perc = config.perc_label
        self.config = config

    def select_samples(self, samples):
        bald_values = BALD(self.config, self.samples)
        _, sorted_index = torch.sort(bald_values, descending=True)
        sample_size = samples.size(0)
        return sorted_index[:int(sample_size * self.config.perc_label)]


def BALD(config: BaseConfig, samples):
    samples_entropy = entropy(config, samples)
    states, actions, rewards, next_states, dones = samples
    num_states = states.size(0)
    batch_size = config.batch_size
    device = config.device
    cond_entropy = torch.zeros((num_states), dtype=torch.float)
    for i in range(0, num_states, batch_size):
        batch_len = min(batch_size, num_states - i)
        batch_states = states[i: i + batch_len, :, :].to(device)
        with torch.no_grad():
            q_values = config.policy_net.generate_q_values(batch_states)
            num_weights = q_values.size(2)
            policy = to_policy(config, q_values)
            current_cond_entropy = torch.sum(policy * torch.log(policy + 1e-8),
                                             dim=2) / num_weights
            cond_entropy[i: i + batch_len] = current_cond_entropy.to('cpu')
    return samples_entropy + cond_entropy
