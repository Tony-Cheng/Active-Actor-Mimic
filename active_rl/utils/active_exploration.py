import random
import torch
import torch.nn.functional as F


class ActionSelector1(object):
    def __init__(self, policy_net, expert_nets, n_steps, same_act_weight,
                 device):
        self._policy_net = policy_net
        self._device = device
        self.n_steps = n_steps
        self.use_expert = False
        self.same_act_weight = same_act_weight
        self.expert_nets = expert_nets
        self.cur_net = policy_net
        self.cur_step = 0
        self.same_actions = []
        for i in range(len(self.expert_nets)):
            self.same_actions.append(0)

        self.reset()

    def select_action(self, state, training=True):
        with torch.no_grad():
            a = self.cur_net(state.to(self._device)).max(1)[1].cpu().view(1, 1)

        for i in range(len(self.expert_nets)):
            expert_a = self.expert_nets[i](state.to(self._device)).max(1)[
                1].cpu().view(1, 1)
            if a == expert_a:
                self.same_actions[i] += 1

        self.cur_step += 1
        if self.cur_step == self.n_steps:
            self.reset(state=state)
        return a.numpy()[0, 0].item()

    def reset(self, state=None):
        self.use_expert = not self.use_expert
        if not self.use_expert:
            self.cur_net = self._policy_net
        elif self.use_expert and state is None:
            net_idx = int(random.random() * len(self.expert_nets))
            self.cur_net = self.expert_nets[net_idx]
        else:
            max_score = -10000
            net_idx = 0
            state = state.to(self._device)
            policy_action_dist = to_prob(self._policy_net(state).squeeze())
            for i in range(len(self.same_actions)):
                action_score = self.same_actions[i] / (1.0 * self.n_steps)
                expert_action_dist = to_prob(
                    self.expert_nets[i](state).squeeze())
                kl_div = F.kl_div(expert_action_dist.log(),
                                  policy_action_dist, reduction='sum')
                score = self.same_act_weight * action_score + \
                    (1. - self.same_act_weight) * kl_div
                if score > max_score:
                    max_score = score
                    net_idx = i

            self.cur_net = self.expert_nets[net_idx]

        for i in range(len(self.same_actions)):
            self.same_actions[i] = 0

        self.cur_step = 0


def to_prob(dist):
    return torch.softmax(dist, dim=0)
