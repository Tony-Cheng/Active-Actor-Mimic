import torch
import torch.nn.functional as F


def standard_optimization(policy_net, target_net, optimizer, memory,
                          batch_size=128, GAMMA=0.99, device='cuda'):

    if len(memory) < batch_size:
        return 0

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    q = policy_net(states).gather(1, actions)
    nq = target_net(next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (
        nq * GAMMA)*(1.-dones[:, 0]) + rewards[:, 0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.detach()


def standard_DDQN(policy_net, target_net, optimizer, memory,
                  batch_size=128, GAMMA=0.99, device='cuda'):

    if len(memory) < batch_size:
        return 0

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    batch_len = states.size(0)

    q = policy_net(states).gather(1, actions)
    next_actions = policy_net(next_states).max(1)[1].view(batch_len, 1)
    nq = target_net(next_states).gather(1, next_actions).detach()

    # Compute the expected Q values
    expected_state_action_values = (
        nq * GAMMA)*(1.-dones) + rewards

    # Compute Huber loss
    loss = F.smooth_l1_loss(q, expected_state_action_values)

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
                      beta=0.01, GAMMA=0.99, training=True, clipping=False):
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
    if clipping:
        for param in AMN_net.parameters():
            param.grad.data.clamp(-1, 1)
    optimizer.step()

    return loss.detach()


# def AMN_perc_optimization(AMN_net, expert_net, optimizer, memory, feature_regression=False, tau=0.1,
#                           percentage=0.1, beta=0.01, batch_size=256, GAMMA=0.99, training=True,
#                           device='cuda'):
#     if not training:
#         return None

#     bs, _, _, _, _ = memory.sample(percentage)
#     bs_len = bs.shape[0]
#     loss = 0
#     for i in range(0, bs_len, batch_size):
#         if i + batch_size < bs_len:
#             actual_batch_size = batch_size
#         else:
#             actual_batch_size = bs_len - i
#         next_bs = bs[i: i + actual_batch_size].to(device)
#         loss += _AMN_optimization(AMN_net, expert_net, optimizer, next_bs,
#                                   tau=tau, beta=beta, GAMMA=GAMMA)
#     return loss


def _AMN_optimization_ENS(AMN_net, expert_net, optimizer, state_batch,
                          ens_num=None, GAMMA=0.99):
    """
    Apply the standard procedure to deep Q network.
    """
    AMN_q_value = AMN_net(state_batch, ens_num=ens_num)
    expert_q_value = expert_net(state_batch)

    AMN_policy = to_policy(AMN_q_value)
    expert_policy = to_policy(expert_q_value).detach()

    loss = -torch.sum(expert_policy * torch.log(AMN_policy + 1e-8))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()


# def AMN_perc_optimization_ENS(AMN_net, expert_net, optimizer, memory,
#                               feature_regression=False, tau=0.1, percentage=0.1,
#                               beta=0.01, batch_size=256, GAMMA=0.99, training=True,
#                               device='cuda'):
#     if not training:
#         return None

#     bs, _, _, _, _ = memory.sample(percentage)
#     bs_len = bs.shape[0]
#     loss = 0
#     for i in range(0, bs_len, batch_size):
#         for ens_num in range(AMN_net.get_num_ensembles()):
#             if i + batch_size < bs_len:
#                 actual_batch_size = batch_size
#             else:
#                 actual_batch_size = bs_len - i
#             next_bs = bs[i: i + actual_batch_size].to(device)
#             loss += _AMN_optimization_ENS(AMN_net, expert_net, optimizer, next_bs, ens_num=ens_num, feature_regression=feature_regression,
#                                           tau=tau, beta=beta, GAMMA=GAMMA, training=training)
#     return loss


# def AMN_optimization_ensemble_epochs(AMN_net, expert_net, optimizer, memory, epochs,
#                                      batch_size=128, GAMMA=0.99, device='cuda'):
#     loss = 0
#     for _ in range(epochs):
#         bs, _, _, _, _ = memory.sample()
#         bs_len = bs.shape[0]
#         for i in range(0, bs_len, batch_size):
#             for ens_num in range(AMN_net.get_num_ensembles()):
#                 if i + batch_size < bs_len:
#                     actual_batch_size = batch_size
#                 else:
#                     actual_batch_size = bs_len - i
#                 next_bs = bs[i: i + actual_batch_size].to(device)
#                 loss += _AMN_optimization_ENS(AMN_net, expert_net,
#                                               optimizer, next_bs,
#                                               ens_num=ens_num,  GAMMA=GAMMA)
#     return loss


def AMN_optimization_ensemble(AMN_net, expert_net, optimizer, memory,
                              batch_size=128, GAMMA=0.99, device='cuda'):
    loss = 0
    bs, _, _, _, _ = memory.sample(batch_size=batch_size)
    bs = bs.to(device)
    for ens_num in range(AMN_net.get_num_ensembles()):
        loss += _AMN_optimization_ENS(AMN_net, expert_net,
                                      optimizer, bs, ens_num=ens_num,
                                      GAMMA=GAMMA)
    return loss / AMN_net.get_num_ensembles()


# def AMN_optimization_epochs(AMN_net, expert_net, optimizer, memory, epochs,
#                             batch_size=128, GAMMA=0.99, device='cuda'):
#     loss = 0
#     for _ in range(epochs):
#         bs, _, _, _, _ = memory.sample()
#         bs_len = bs.shape[0]
#         for i in range(0, bs_len, batch_size):
#             if i + batch_size < bs_len:
#                 actual_batch_size = batch_size
#             else:
#                 actual_batch_size = bs_len - i
#             next_bs = bs[i: i + actual_batch_size].to(device)
#             loss += _AMN_optimization(AMN_net, expert_net,
#                                       optimizer, next_bs, GAMMA=GAMMA)
#     return loss


def standard_optimization_ensemble(policy_net, target_net, optimizer, memory,
                                   batch_size=128, GAMMA=0.99, device='cuda'):
    """
    Apply the standard procedure to an ensemble of deep Q network.
    """
    if len(memory) < batch_size:
        return 0

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    total_loss = 0
    for ens_num in range(policy_net.get_num_ensembles()):
        q = policy_net(states, ens_num=ens_num).gather(1, actions)
        nq = target_net(next_states, ens_num=ens_num).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (
            nq * GAMMA)*(1.-dones[:, 0]) + rewards[:, 0]

        # Compute Huber loss
        loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()

    return total_loss / policy_net.get_num_ensembles()


def standard_opt_ens_ddqn(policy_net, target_net, optimizer, memory,
                          batch_size=128, GAMMA=0.99, device='cuda'):
    """
    Apply the standard procedure to an ensemble of deep Q network.
    """
    if len(memory) < batch_size:
        return 0

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    batch_len = states.size(0)

    total_loss = 0
    for ens_num in range(policy_net.get_num_ensembles()):
        q = policy_net(states, ens_num=ens_num).gather(1, actions)
        next_actions = policy_net(next_states, ens_num=ens_num).max(1)[
            1].view(batch_len, 1)
        nq = target_net(next_states, ens_num=ens_num).gather(
            1, next_actions).detach()

        # Compute the expected Q values
        expected_state_action_values = (
            nq * GAMMA)*(1.-dones) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(q, expected_state_action_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()

    return total_loss / policy_net.get_num_ensembles()


def to_policy(q_values, tau=0.1):
    return F.softmax(q_values / tau, dim=1)
