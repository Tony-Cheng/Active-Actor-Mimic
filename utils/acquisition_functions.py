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
        current_entropy = - torch.sum(policy * torch.log(policy), 1)
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
                                                  torch.log(policy), 1)
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
                actions[:,j] = torch.argmax(q_values, dim=1)
        modal = torch.mode(actions, dim=1)[0].unsqueeze(1)
        frequency = torch.sum(actions == modal, 1)
        current_var_ration = 1 - frequency / num_iters
        var_ratio[i : i + batch_len] = current_var_ration
    return var_ratio

def mc_random(policy_net, states, tau=0.1, batch_size=128, num_iters=10, device='cuda'):
    return torch.randn((states.shape[0]), dtype=torch.float)

def to_policy(q_values, tau=0.1):
    return F.softmax(q_values / tau, dim=1)
