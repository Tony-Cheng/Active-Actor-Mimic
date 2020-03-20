import random
from .config import DiscreteActionConfig
import torch


class DiscreteActionSelector:
    def __init__(self, config: DiscreteActionConfig):
        self._eps = config.eps_start
        self._FINAL_EPSILON = config.eps_end
        self._INITIAL_EPSILON = config.eps_start
        self._policy_net = config.policy_net
        self._EPS_DECAY = config.eps_decay
        self._n_actions = config.env.n_actions
        self._device = config.device

    def select_action(self, state, training=True):
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

        return a.numpy()[0, 0].item()
