import torch
from collections import namedtuple
import torch.nn.functional as F


def standard_optimization(policy_net, target_net, optimizer, memory, batch_size=128,
                          GAMMA=0.99, training=True):
    if not training:
        return None
    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(
        batch_size)

    q = policy_net(state_batch).gather(1, action_batch)
    nq = target_net(n_state_batch).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (
        nq * GAMMA)*(1.-done_batch[:, 0]) + reward_batch[:, 0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.detach()


def AMN_optimization(AMN_net, expert_net, optimizer, memory, feature_regression=False, tau=0.1,
                     beta=0.01, batch_size=256, GAMMA=0.99, training=True):
    """
    Apply the standard procedure to deep Q network.
    """
    if not training:
        return None
    state_batch, _, _, _, _ = memory.sample(batch_size)

    if feature_regression == True:
        AMN_q_value, AMN_last_layer = AMN_net(state_batch, last_layer=True)
        expert_q_value, expert_last_layer = expert_net(
            state_batch, last_layer=True)
        loss = F.mse_loss(AMN_last_layer, expert_last_layer.detach())
    else:
        AMN_q_value = AMN_net(state_batch, last_layer=False)
        expert_q_value = expert_net(state_batch, last_layer=False)
        loss = 0

    AMN_policy = to_policy(AMN_q_value)
    expert_policy = to_policy(expert_q_value).detach()

    loss -= torch.sum(expert_policy * torch.log(AMN_policy + 1e-8))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()


def _AMN_optimization(AMN_net, expert_net, optimizer, state_batch, feature_regression=False, tau=0.1,
                      beta=0.01, GAMMA=0.99, training=True):
    """
    Apply the standard procedure to deep Q network.
    """
    if not training:
        return None

    if feature_regression == True:
        AMN_q_value, AMN_last_layer = AMN_net(state_batch, last_layer=True)
        expert_q_value, expert_last_layer = expert_net(
            state_batch, last_layer=True)
        loss = F.mse_loss(AMN_last_layer, expert_last_layer.detach())
    else:
        AMN_q_value = AMN_net(state_batch, last_layer=False)
        expert_q_value = expert_net(state_batch, last_layer=False)
        loss = 0

    AMN_policy = to_policy(AMN_q_value)
    expert_policy = to_policy(expert_q_value).detach()

    loss -= torch.sum(expert_policy * torch.log(AMN_policy + 1e-8))

    optimizer.zero_grad()
    loss.backward()
    for param in AMN_net.parameters():
        param.grad.data.clamp(-1, 1)
    optimizer.step()

    return loss.detach()


def AMN_perc_optimization(AMN_net, expert_net, optimizer, memory, feature_regression=False, tau=0.1,
                          percentage=0.1, beta=0.01, batch_size=256, GAMMA=0.99, training=True,
                          device='cuda'):
    if not training:
        return None

    bs, _, _, _, _ = memory.sample(percentage)
    bs_len = bs.shape[0]
    loss = 0
    for i in range(0, bs_len, batch_size):
        if i + batch_size < bs_len:
            actual_batch_size = batch_size
        else:
            actual_batch_size = bs_len - i
        next_bs = bs[i: i + actual_batch_size].to(device)
        loss += _AMN_optimization(AMN_net, expert_net, optimizer, next_bs, feature_regression=feature_regression,
                                  tau=tau, beta=beta, GAMMA=GAMMA, training=training)
    return loss


def _AMN_optimization_ENS(AMN_net, expert_net, optimizer, state_batch, ens_num=None, feature_regression=False, tau=0.1,
                          beta=0.01, GAMMA=0.99, training=True):
    """
    Apply the standard procedure to deep Q network.
    """
    if not training:
        return None

    if feature_regression == True:
        AMN_q_value, AMN_last_layer = AMN_net(state_batch, last_layer=True)
        expert_q_value, expert_last_layer = expert_net(
            state_batch, last_layer=True)
        loss = F.mse_loss(AMN_last_layer, expert_last_layer.detach())
    else:
        AMN_q_value = AMN_net(state_batch, ens_num=ens_num, last_layer=False)
        expert_q_value = expert_net(state_batch, last_layer=False)
        loss = 0

    AMN_policy = to_policy(AMN_q_value)
    expert_policy = to_policy(expert_q_value).detach()

    loss -= torch.sum(expert_policy * torch.log(AMN_policy + 1e-8))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()


def AMN_perc_optimization_ENS(AMN_net, expert_net, optimizer, memory, feature_regression=False, tau=0.1,
                              percentage=0.1, beta=0.01, batch_size=256, GAMMA=0.99, training=True,
                              device='cuda'):
    if not training:
        return None

    bs, _, _, _, _ = memory.sample(percentage)
    bs_len = bs.shape[0]
    loss = 0
    for i in range(0, bs_len, batch_size):
        for ens_num in range(AMN_net.get_num_ensembles()):
            if i + batch_size < bs_len:
                actual_batch_size = batch_size
            else:
                actual_batch_size = bs_len - i
            next_bs = bs[i: i + actual_batch_size].to(device)
            loss += _AMN_optimization_ENS(AMN_net, expert_net, optimizer, next_bs, ens_num=ens_num, feature_regression=feature_regression,
                                          tau=tau, beta=beta, GAMMA=GAMMA, training=training)
    return loss


def AMN_optimization_ensemble_epochs(AMN_net, expert_net, optimizer, memory, epochs,
                                     batch_size=128, GAMMA=0.99, device='cuda'):
    loss = 0
    for _ in range(epochs):
        bs, _, _, _, _ = memory.sample()
        bs_len = bs.shape[0]
        for i in range(0, bs_len, batch_size):
            for ens_num in range(AMN_net.get_num_ensembles()):
                if i + batch_size < bs_len:
                    actual_batch_size = batch_size
                else:
                    actual_batch_size = bs_len - i
                next_bs = bs[i: i + actual_batch_size].to(device)
                loss += _AMN_optimization_ENS(AMN_net, expert_net,
                                              optimizer, next_bs, ens_num=ens_num,  GAMMA=GAMMA)
    return loss


def AMN_optimization_epochs(AMN_net, expert_net, optimizer, memory, epochs,
                            batch_size=128, GAMMA=0.99, device='cuda'):
    loss = 0
    for _ in range(epochs):
        bs, _, _, _, _ = memory.sample()
        bs_len = bs.shape[0]
        for i in range(0, bs_len, batch_size):
            if i + batch_size < bs_len:
                actual_batch_size = batch_size
            else:
                actual_batch_size = bs_len - i
            next_bs = bs[i: i + actual_batch_size].to(device)
            loss += _AMN_optimization_ENS(AMN_net,
                                          expert_net, optimizer, next_bs, GAMMA=GAMMA)
    return loss


def standard_optimization_ensemble(policy_net, target_net, optimizer, memory, batch_size=128,
                                   GAMMA=0.99, device='cuda'):
    """
    Apply the standard procedure to an ensemble of deep Q network.
    """
    if len(memory) < batch_size:
        return 0

    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(
        batch_size)

    state_batch = state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    n_state_batch = n_state_batch.to(device)
    done_batch = done_batch.to(device)

    total_loss = 0
    for ens_num in range(policy_net.get_num_ensembles()):
        q = policy_net(state_batch, ens_num=ens_num).gather(1, action_batch)
        nq = target_net(n_state_batch, ens_num=ens_num).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (
            nq * GAMMA)*(1.-done_batch[:, 0]) + reward_batch[:, 0]

        # Compute Huber loss
        loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()

    return total_loss


def to_policy(q_values, tau=0.1):
    return F.softmax(q_values / tau, dim=1)
