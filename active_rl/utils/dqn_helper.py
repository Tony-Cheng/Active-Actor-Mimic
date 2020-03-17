import torch.functional as F


def to_policy(config, q_values):
    return F.softmax(q_values / config.tau, dim=1)