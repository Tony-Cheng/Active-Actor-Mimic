import numpy as np
import torch
import random
import math


def get_state(obs):
    """
    Convert an open AI gym state to numpy state.
    """
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
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
