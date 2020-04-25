import torch


def compute_q_n2s_ratio(dqns, states, actions, device='cuda'):
    states = states.to(device)
    actions = actions.to(device)
    q_vals = []
    num_nets = len(dqns)
    for dqn in dqns:
        with torch.no_grad():
            q_vals.append(dqn(states))
    q_vals = torch.stack(q_vals)
    expected_q_vals = torch.sum(q_vals, dim=0) / num_nets
    sample_var = torch.sum((q_vals - expected_q_vals)
                           ** 2, dim=0) / (num_nets - 1)
    n2s_ratio = sample_var / (expected_q_vals ** 2 + 1e-6)
    action_n2s = n2s_ratio.gather(1, actions).squeeze()
    batch_len = states.size(0)
    action_len = q_vals.size(2)
    not_action_mask = torch.ones((batch_len, action_len)).to(device)
    zero_mask = torch.zeros((batch_len, 1)).to(device)
    not_action_mask = not_action_mask.scatter_(
        dim=1, index=actions, src=zero_mask)
    not_action_n2s = (n2s_ratio * not_action_mask)
    avg_not_action_n2s = torch.sum(not_action_n2s, dim=1) / (action_len - 1)
    action_n2s = (torch.sum(action_n2s) / batch_len).cpu()
    avg_not_action_n2s = (torch.sum(avg_not_action_n2s) / batch_len).cpu()
    not_action_to_action_ratio = avg_not_action_n2s / action_n2s
    return action_n2s, avg_not_action_n2s, not_action_to_action_ratio


def compute_q_s2n_ratio(dqns, states, actions, device='cuda'):
    states = states.to(device)
    actions = actions.to(device)
    q_vals = []
    num_nets = len(dqns)
    for dqn in dqns:
        with torch.no_grad():
            q_vals.append(dqn(states))
    q_vals = torch.stack(q_vals)
    expected_q_vals = torch.sum(q_vals, dim=0) / num_nets
    sample_var = torch.sum((q_vals - expected_q_vals)
                           ** 2, dim=0) / (num_nets - 1)
    s2n_ratio = (expected_q_vals ** 2) / (sample_var + + 1e-6)
    action_s2n = s2n_ratio.gather(1, actions).squeeze()
    batch_len = states.size(0)
    action_len = q_vals.size(2)
    not_action_mask = torch.ones((batch_len, action_len)).to(device)
    zero_mask = torch.zeros((batch_len, 1)).to(device)
    not_action_mask = not_action_mask.scatter_(
        dim=1, index=actions, src=zero_mask)
    not_action_s2n = (s2n_ratio * not_action_mask)
    avg_not_action_s2n = torch.sum(not_action_s2n, dim=1) / (action_len - 1)
    action_s2n = (torch.sum(action_s2n) / batch_len).cpu()
    avg_not_action_s2n = (torch.sum(avg_not_action_s2n) / batch_len).cpu()
    not_action_to_action_ratio = avg_not_action_s2n / action_s2n
    return action_s2n, avg_not_action_s2n, not_action_to_action_ratio
