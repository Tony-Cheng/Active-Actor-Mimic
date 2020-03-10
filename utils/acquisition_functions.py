import torch
import torch.nn.functional as F


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
        next_states = states[i: i + batch_len, :, :].to(device)
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
        next_states = states[i: i + batch_len, :, :].to(device)
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


def ens_TD_no_target(policy_net, memory, batch_size=128, GAMMA=0.99, device='cuda'):
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
            loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))
            td_loss[i: i + batch_len] = loss.to('cpu')
    return - td_loss


def ens_random(policy_net, states, tau=0.1, batch_size=128, device='cuda'):
    return torch.randn((states.shape[0]), dtype=torch.float)
