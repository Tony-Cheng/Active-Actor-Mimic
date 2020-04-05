import torch
from torch.distributions.dirichlet import Dirichlet
import random


class UniformActionSelector(object):
    def __init__(self, INITIAL_EPSILON, FINAL_EPSILON, policy_net, EPS_DECAY,
                 n_actions, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device
        self._cur_ens_num = 0

    def select_action(self, state, training=True):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps:
            with torch.no_grad():
                if training:
                    a = self._policy_net(state.to(self._device),
                                         ens_num=self._cur_ens_num).max(1)[
                        1].cpu().view(1, 1)
                else:
                    a = self._policy_net(state.to(self._device)).max(1)[
                        1].cpu().view(1, 1)
        else:
            a = torch.tensor([[random.randrange(self._n_actions)]],
                             device='cpu', dtype=torch.long)

        return a.numpy()[0, 0].item(), self._eps

    def reset_ens_num(self):
        self._cur_ens_num = int(
            random.random() * self._policy_net.get_num_ensembles())


class DirichletActionSelector(object):
    def __init__(self, INITIAL_EPSILON, FINAL_EPSILON, policy_net, EPS_DECAY,
                 n_actions, lamb, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device
        distn_params = [
            1 / lamb for _ in range(policy_net.get_num_ensembles())]
        self.distn = Dirichlet(torch.tensor(distn_params))

    def select_action(self, state, training=True):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps:
            with torch.no_grad():
                if training:
                    alpha = self.distn.sample()
                    q_val = 0
                    for i in range(self._policy_net.get_num_ensembles()):
                        q_val += alpha[i] * \
                            self._policy_net(state.to(self._device), ens_num=i)
                    a = q_val.max(1)[1].cpu().view(1, 1)
                else:
                    a = self._policy_net(state.to(self._device)).max(1)[
                        1].cpu().view(1, 1)
        else:
            a = torch.tensor([[random.randrange(self._n_actions)]],
                             device='cpu', dtype=torch.long)

        return a.numpy()[0, 0].item(), self._eps
