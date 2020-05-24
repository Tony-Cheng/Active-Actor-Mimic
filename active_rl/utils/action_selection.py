import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial
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


class ArgmaxActionSelector(object):
    def __init__(self, INITIAL_EPSILON, FINAL_EPSILON, policy_net, EPS_DECAY,
                 n_actions, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device

    def select_action(self, state, training=True):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps:
            opt_a = None
            opt_v = None
            for i in range(self._policy_net.get_num_ensembles()):
                with torch.no_grad():
                    q_val = self._policy_net(state.to(self._device)).max(1)
                    a = q_val[1].cpu().view(1, 1)
                    v = q_val[0].cpu().view(1, 1).numpy()[0, 0].item()
                    if opt_v is None or v > opt_v:
                        opt_v = v
                        opt_a = a
        else:
            opt_a = torch.tensor([[random.randrange(self._n_actions)]],
                                 device='cpu', dtype=torch.long)

        return opt_a.numpy()[0, 0].item(), self._eps


class DuoActionSelector(object):
    def __init__(self, INITIAL_EPSILON, FINAL_EPSILON, policy_net1,
                 policy_net2, EPS_DECAY, n_actions, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net1 = policy_net1
        self._policy_net2 = policy_net2
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device

    def select_action(self, state, training=True):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps:
            with torch.no_grad():
                a = self._policy_net1(state.to(self._device)).max(1)[
                    1].cpu().view(1, 1)
        else:
            with torch.no_grad():
                a = self._policy_net2(state.to(self._device)).max(1)[
                    1].cpu().view(1, 1)

        return a.numpy()[0, 0].item(), self._eps


class AdvancedActionSelector(object):
    def __init__(self, policy_net, INITIAL_EPSILON, FINAL_EPSILON, EPS_DECAY,
                 tau, n_actions, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device
        self._tau = tau

    def select_action(self, state, training=True):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps:
            with torch.no_grad():
                state = state.to(self._device)
                a = self._policy_net(state, tau=self._tau).max(1)[
                    1].cpu().view(1, 1)
        else:
            a = torch.tensor([[random.randrange(self._n_actions)]],
                             device='cpu', dtype=torch.long)

        return a.numpy()[0, 0].item(), self._eps


class ProbabilisticActionSelector(object):
    def __init__(self, policy_net, INITIAL_EPSILON, FINAL_EPSILON, EPS_DECAY,
                 n_actions, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device
        self._dist = Normal(0, 1)

    def select_action(self, state, training=True):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        action_mean = None
        action_var = None
        if sample > self._eps:
            with torch.no_grad():
                q_vals = torch.zeros((self._policy_net.get_num_ensembles(),
                                      self._n_actions))
                state = state.to(self._device)
                for i in range(self._policy_net.get_num_ensembles()):
                    q_vals[i, :] = self._policy_net(
                        state, ens_num=i).to('cpu').squeeze(0)
                action_mean = torch.mean(q_vals, 0)
                action_var = torch.var(q_vals, 0)
                top_idx = torch.argmax(action_mean)
                score = torch.zeros((self._n_actions))
                for i in range(self._n_actions):
                    normal_val = (action_mean[top_idx] - action_mean[i]) / \
                        (action_var[top_idx] + action_var[i])
                    score[i] = 1. - self._dist.cdf(normal_val)
                action_dist = Multinomial(1, score)
                a = action_dist.sample().argmax().item()
        else:
            a = torch.tensor([[random.randrange(self._n_actions)]],
                             device='cpu', dtype=torch.long).numpy()[0, 0].item()

        return a, self._eps, action_mean, action_var


class MeanActionSelector(object):
    def __init__(self, policy_net, INITIAL_EPSILON, FINAL_EPSILON, EPS_DECAY,
                 n_actions, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device
        self._dist = Normal(0, 1)

    def select_action(self, state, training=True):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON -
                          self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        action_mean = None
        if sample > self._eps:
            with torch.no_grad():
                q_vals = torch.zeros((self._policy_net.get_num_ensembles(),
                                      self._n_actions))
                state = state.to(self._device)
                for i in range(self._policy_net.get_num_ensembles()):
                    q_vals[i, :] = self._policy_net(
                        state, ens_num=i).to('cpu').squeeze(0)
                action_mean = torch.mean(q_vals, 0)
                a = action_mean.argmax().item()
        else:
            a = torch.tensor([random.randrange(self._n_actions)],
                             device='cpu', dtype=torch.long).numpy()[0].item()

        return a, self._eps, action_mean
