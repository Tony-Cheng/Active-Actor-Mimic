import torch
import torch.nn.functional as F


def BALD(prob_N_K_C):
    prob_N_C = torch.mean(prob_N_K_C, dim=1)
    entropy = torch.sum(prob_N_C * torch.log(prob_N_C + 1e-7), dim=1)
    cond_entropy = torch.mean(
        torch.sum(prob_N_K_C * torch.log(prob_N_K_C + 1e-7), dim=2), dim=1)
    return entropy - cond_entropy


def mc_entropy(policy_net, states, tau=0.1, batch_size=128, num_iters=10, device='cuda'):
    entropy = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :, :].to(device)
        with torch.no_grad():
            q_values = policy_net(next_states)
            policy = to_policy(q_values, tau=tau)
        for _ in range(num_iters - 1):
            with torch.no_grad():
                q_values = policy_net(next_states)
                policy += to_policy(q_values, tau=tau)
        policy /= num_iters
        current_entropy = - torch.sum(policy * torch.log(policy + 1e-8), 1)
        entropy[i: i + batch_len] = current_entropy.to('cpu')
    return entropy


def mc_BALD(policy_net, states, tau=0.1, batch_size=128, num_iters=10, device='cuda'):
    entropy = mc_entropy(policy_net, states, tau=tau,
                         batch_size=batch_size, num_iters=num_iters, device=device)
    cond_entropy = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :, :].to(device)
        with torch.no_grad():
            q_values = policy_net(next_states)
            policy = to_policy(q_values, tau=tau)
            current_cond_entropy = torch.sum(policy * torch.log(policy), 1)
        for _ in range(num_iters - 1):
            with torch.no_grad():
                q_values = policy_net(next_states)
                policy = to_policy(q_values, tau=tau)
                current_cond_entropy += torch.sum(policy *
                                                  torch.log(policy + 1e-8), 1)
        current_cond_entropy /= num_iters
        cond_entropy[i: i + batch_len] = current_cond_entropy.to('cpu')
    return entropy + cond_entropy


def mc_var_ratio(policy_net, states, tau=0.1, batch_size=128, num_iters=10, device='cuda'):
    var_ratio = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :, :].to(device)
        actions = torch.zeros((next_states.shape[0], num_iters))
        for j in range(num_iters - 1):
            with torch.no_grad():
                q_values = policy_net(next_states)
                actions[:, j] = torch.argmax(q_values, dim=1)
        modal = torch.mode(actions, dim=1)[0].unsqueeze(1)
        frequency = torch.sum(actions == modal, 1)
        current_var_ration = 1 - frequency / num_iters
        var_ratio[i: i + batch_len] = current_var_ration
    return var_ratio


def mc_random(policy_net, states, tau=0.1, batch_size=128, num_iters=10, device='cuda'):
    return torch.randn((states.shape[0]), dtype=torch.float)


def to_policy(q_values, tau=0.1):
    return F.softmax(q_values / tau, dim=1)


def ens_entropy(policy_net, states, tau=0.1, batch_size=128, device='cuda'):
    entropy = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :4, :].to(device)
        with torch.no_grad():
            q_values = policy_net(next_states, ens_num=0)
            policy = to_policy(q_values, tau=tau)
        for j in range(1, policy_net.get_num_ensembles()):
            with torch.no_grad():
                q_values = policy_net(next_states, ens_num=j)
                policy += to_policy(q_values, tau=tau)
        policy /= policy_net.get_num_ensembles()
        current_entropy = - torch.sum(policy * torch.log(policy + 1e-8), 1)
        entropy[i: i + batch_len] = current_entropy.to('cpu')
    return entropy


def ens_BALD(policy_net, states, tau=0.1, batch_size=128, device='cuda'):
    entropy = ens_entropy(policy_net, states, tau=tau,
                          batch_size=batch_size, device=device)
    cond_entropy = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :4].to(device)
        with torch.no_grad():
            q_values = policy_net(next_states, ens_num=0)
            policy = to_policy(q_values, tau=tau)
            current_cond_entropy = torch.sum(policy * torch.log(policy), 1)
        for j in range(1, policy_net.get_num_ensembles()):
            with torch.no_grad():
                q_values = policy_net(next_states, ens_num=j)
                policy = to_policy(q_values, tau=tau)
                current_cond_entropy += torch.sum(policy *
                                                  torch.log(policy + 1e-8), 1)
        current_cond_entropy /= policy_net.get_num_ensembles()
        cond_entropy[i: i + batch_len] = current_cond_entropy.to('cpu')
    return entropy + cond_entropy


def ens_normalized_BALD(policy_net, states, tau=0.1, batch_size=128, device='cuda'):
    BALD_vals = ens_BALD(policy_net, states, tau=0.1,
                         batch_size=128, device=device)
    min_BALD = torch.min(BALD_vals)
    max_BALD = torch.max(BALD_vals)
    return (BALD_vals - min_BALD) / (max_BALD - min_BALD + 1e-9)


def ens_value(value_net, states, batch_size=128, device='cuda'):
    values = torch.empty((states.shape[0]))
    num_states = states.shape[0]
    for i in range(0, num_states, batch_size):
        next_batch_size = batch_size
        if (num_states - i < batch_size):
            next_batch_size = num_states - i
        next_states = states[i:i + next_batch_size, :4].to(device)
        with torch.no_grad():
            values[i:i + next_batch_size] = value_net(next_states).squeeze()
    return values


def dqn_normalized_value(value_net, states, batch_size=128, device='cuda'):
    values = torch.empty((states.shape[0]))
    num_states = states.shape[0]
    with torch.no_grad():
        for i in range(0, num_states, batch_size):
            next_batch_size = batch_size
            if (num_states - i < batch_size):
                next_batch_size = num_states - i
            next_states = states[i:i + next_batch_size, :4].to(device)
            values[i:i + next_batch_size] = value_net(next_states).max(dim=1)[0]
    min_value = torch.min(values)
    max_value = torch.max(values)
    return (values - min_value) / (max_value - min_value + 1e-9)


def mixed_BALD_value(AMN_net, value_net, AMN_weight, batch_size, device):
    return lambda states: AMN_weight * ens_normalized_BALD(AMN_net, states, batch_size=batch_size, device=device) + \
        (1 - AMN_weight) * dqn_normalized_value(value_net, states, batch_size=batch_size, device=device)


def ens_value_td(value_net, states, batch_size=128, device='cuda'):
    td_values = torch.empty((states.shape[0]))
    num_states = states.shape[0]
    for i in range(0, num_states, batch_size):
        next_batch_size = batch_size
        if (num_states - i < batch_size):
            next_batch_size = num_states - i
        cur_states = states[i:i + next_batch_size, :4].to(device)
        next_states = states[i:i + next_batch_size, 1:].to(device)
        with torch.no_grad():
            td_values[i:i + next_batch_size] = torch.abs(value_net(next_states).squeeze() -
                                                         value_net(cur_states).squeeze())
    return td_values


def ens_BALD_prob(policy_net, obs, tau=0.1, batch_size=128, device='cuda'):
    states, actions, _, _, _ = obs
    BALD_val = ens_BALD(policy_net, states, tau=tau, batch_size=batch_size,
                        device=device)
    prob = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :, :].to(device)
        next_actions = actions[i: i + batch_len, :].to(device)
        total_policy = 0
        for j in range(policy_net.get_num_ensembles()):
            with torch.no_grad():
                q_values = policy_net(next_states, ens_num=j)
                policy = to_policy(q_values, tau=tau).gather(1, next_actions)
                total_policy += policy.squeeze()
        total_policy /= policy_net.get_num_ensembles()
        prob[i: i + batch_len] = total_policy.to('cpu')
    return BALD_val * prob


def ens_BALD_prob_sum(policy_net, obs, tau=0.1, batch_size=128, device='cuda'):
    states, actions, _, _, _ = obs
    BALD_val = ens_BALD(policy_net, states, tau=tau, batch_size=batch_size,
                        device=device)
    prob = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :, :].to(device)
        next_actions = actions[i: i + batch_len, :].to(device)
        total_policy = 0
        for j in range(policy_net.get_num_ensembles()):
            with torch.no_grad():
                q_values = policy_net(next_states, ens_num=j)
                policy = to_policy(q_values, tau=tau).gather(1, next_actions)
                total_policy += policy.squeeze()
        total_policy /= policy_net.get_num_ensembles()
        prob[i: i + batch_len] = total_policy.to('cpu')
    return BALD_val + prob


def ens_BALD_inv_prob(policy_net, obs, tau=0.1, batch_size=128, device='cuda'):
    states, actions, _, _, _ = obs
    BALD_val = ens_BALD(policy_net, states, tau=tau, batch_size=batch_size,
                        device=device)
    prob = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :, :].to(device)
        next_actions = actions[i: i + batch_len, :].to(device)
        total_policy = 0
        for j in range(policy_net.get_num_ensembles()):
            with torch.no_grad():
                q_values = policy_net(next_states, ens_num=j)
                policy = to_policy(q_values, tau=tau).gather(1, next_actions)
                total_policy += policy.squeeze()
        total_policy /= policy_net.get_num_ensembles()
        prob[i: i + batch_len] = total_policy.to('cpu')
    return BALD_val / (prob + 1e-7)


def ens_TD_no_target(policy_net, memory, batch_size=128, GAMMA=0.99,
                     device='cuda'):
    bs, ba, br, bns, bd = memory.get_all()
    td_loss = torch.zeros((bs.shape[0]), dtype=torch.float)
    for i in range(0, bs.shape[0], batch_size):
        if i + batch_size < bs.shape[0]:
            batch_len = batch_size
        else:
            batch_len = bs.shape[0] - i
        state_batch = bs[i: i + batch_len, :, :].to(device)
        n_state_batch = bns[i: i + batch_len, :, :].to(device)
        action_batch = ba[i: i + batch_len, :].to(device)
        reward_batch = br[i: i + batch_len, :].to(device)
        done_batch = bd[i: i + batch_len, :].to(device)
        with torch.no_grad():
            q = policy_net(state_batch).gather(1, action_batch)
            nq = policy_net(n_state_batch).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (
                nq * GAMMA)*(1.-done_batch[:, 0]) + reward_batch[:, 0]
            # Compute Huber loss
            loss = F.smooth_l1_loss(
                q, expected_state_action_values.unsqueeze(1))
            td_loss[i: i + batch_len] = loss.to('cpu')
    return td_loss


def ens_SNR(policy_net, states, tau=0.1, batch_size=128, device='cuda'):
    SNR = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :, :].to(device)
        sum_q = 0
        sum_q_2 = 0
        num_ens = policy_net.get_num_ensembles()
        for j in range(num_ens):
            with torch.no_grad():
                q_values = policy_net(next_states, ens_num=j)
                sum_q += q_values
                sum_q_2 += q_values ** 2
        sample_var = (sum_q_2 - (sum_q ** 2) / num_ens) / (num_ens - 1)
        n_actions = sample_var.size(1)
        cur_SNR = torch.sum(sample_var / ((sum_q / num_ens)
                                          ** 2 + 1e-8), dim=1) / n_actions
        SNR[i: i + batch_len] = cur_SNR.to('cpu')
    return SNR


def ens_N2S(policy_net, states, tau=0.1, batch_size=128, device='cuda'):
    n2s = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :, :].to(device)
        q_vals = []
        num_ens = policy_net.get_num_ensembles()
        for j in range(num_ens):
            with torch.no_grad():
                q_vals.append(policy_net(next_states, ens_num=j))
        q_vals = torch.stack(q_vals)
        expected_q_vals = torch.sum(q_vals, dim=0) / num_ens
        sample_var = torch.sum((q_vals - expected_q_vals)
                               ** 2, dim=0) / (num_ens - 1)
        n2s_ratio = sample_var / (expected_q_vals ** 2 + 1e-6)
        n2s[i: i + batch_len] = torch.sum(n2s_ratio, dim=1).to('cpu')
    return n2s


def ens_n2s_action_gap(policy_net, states, tau=0.1, batch_size=128, device='cuda'):
    n2s = torch.zeros((states.shape[0]), dtype=torch.float)
    for i in range(0, states.shape[0], batch_size):
        if i + batch_size < states.shape[0]:
            batch_len = batch_size
        else:
            batch_len = states.shape[0] - i
        next_states = states[i: i + batch_len, :, :].to(device)
        q_vals = []
        num_ens = policy_net.get_num_ensembles()
        for j in range(num_ens):
            with torch.no_grad():
                q_vals.append(policy_net(next_states, ens_num=j))
        q_vals = torch.stack(q_vals)
        expected_q_vals = torch.sum(q_vals, dim=0) / num_ens
        top_actions = torch.argsort(
            expected_q_vals, dim=1, descending=True)[:, :2]
        a1_q_vals = q_vals.gather(
            2, top_actions[:, 0:1].unsqueeze(0).expand_as(q_vals[:, :, 0:1]))
        a2_q_vals = q_vals.gather(
            2, top_actions[:, 1:2].unsqueeze(0).expand_as(q_vals[:, :, 0:1]))
        q_vals_difference = a1_q_vals - a2_q_vals

        expected_q_vals_difference = torch.sum(
            q_vals_difference, dim=0) / num_ens
        sample_var = torch.sum((q_vals_difference - expected_q_vals_difference)
                               ** 2, dim=0) / (num_ens - 1)
        n2s_ratio = sample_var / (expected_q_vals_difference ** 2 + 1e-6)
        n2s[i: i + batch_len] = torch.sum(n2s_ratio, dim=1).to('cpu')
    return n2s


def ens_random(policy_net, states, tau=0.1, batch_size=128, device='cuda'):
    return torch.randn((states.shape[0]), dtype=torch.float)
