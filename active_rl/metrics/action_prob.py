import torch.nn.functional as F
import torch


def prob_action(policy_net, memory, tau, batch_size, device):
    states, actions, _, _, _ = memory.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)

    policy = 0
    for ens_num in range(policy_net.get_num_ensembles()):
        with torch.no_grad():
            policy += F.softmax(policy_net(states,
                                           ens_num=ens_num) / tau, dim=1)
    policy /= policy_net.get_num_ensembles()

    prob_actions = policy.gather(1, actions)
    return torch.sum(prob_actions) / batch_size


def prob_action_bootstrap(policy_net, memory, tau, batch_size, device):
    states, actions, _, _, _ = memory.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)

    policy = 0
    with torch.no_grad():
        qs = policy_net(states, training=True)

    for i in range(len(qs)):
        q = qs[i]
        policy += F.softmax(q / tau, dim=1)

    policy /= len(qs)

    prob_actions = policy.gather(1, actions)
    return torch.sum(prob_actions) / batch_size
