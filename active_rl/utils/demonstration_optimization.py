import torch.nn.functional as F
import torch


def standard_optimization(policy_net, target_net, memory, optimizer, gamma, lambda1, lambda2, batch_size, device):
    (states, actions, rewards, next_states, dones, rollout_rewards, rollout_offsets,
     rollout_next_states, rollout_dones) = memory.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    rollout_rewards = rollout_rewards.to(device)
    rollout_offsets = rollout_offsets.to(device)
    rollout_next_states = rollout_next_states.to(device)
    rollout_dones = rollout_dones.to(device)

    q = policy_net(states).gather(1, actions)
    nq = target_net(next_states).max(1)[0].detach()

    expected_state_action_values = (
        nq * gamma)*(1.-dones[:, 0]) + rewards[:, 0]

    # Compute Huber loss
    loss_q = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    rollout_nq = target_net(rollout_next_states).max(1)[0].detach()

    expected_rollout_value = rollout_rewards[:, 0] + \
        (1 - rollout_dones[:, 0]) * \
        ((gamma ** rollout_offsets[:, 0]) * rollout_nq)

    loss_rollout = F.smooth_l1_loss(q, expected_rollout_value.unsqueeze(1))

    max_q, argmax_q = policy_net(states).max(1)
    max_q = max_q.unsqueeze(1)
    argmax_q = argmax_q.unsqueeze(1)
    non_identity_value = ((1.-(actions == argmax_q).float()) * 0.01)
    loss_e = torch.sum(max_q + non_identity_value - q)

    total_loss = loss_q + lambda1 * loss_rollout + lambda2 * loss_e

    optimizer.zero_grad()
    total_loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return total_loss.detach()


def demo_optimization_ens_no_mem(policy_net, target_net, samples, optimizer, gamma, lambda1, lambda2, batch_size, device):
    (states, actions, rewards, next_states, dones, rollout_rewards, rollout_offsets,
     rollout_next_states, rollout_dones) = samples

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    rollout_rewards = rollout_rewards.to(device)
    rollout_offsets = rollout_offsets.to(device)
    rollout_next_states = rollout_next_states.to(device)
    rollout_dones = rollout_dones.to(device)

    total_loss = 0
    for ens_num in range(policy_net.get_num_ensembles()):
        q = policy_net(states, ens_num=ens_num).gather(1, actions)
        nq = target_net(next_states, ens_num=ens_num).max(1)[0].detach()

        expected_state_action_values = (
            nq * gamma)*(1.-dones[:, 0]) + rewards[:, 0]

        # Compute Huber loss
        loss_q = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

        rollout_nq = target_net(rollout_next_states, ens_num=ens_num).max(1)[
            0].detach()

        expected_rollout_value = rollout_rewards[:, 0] + \
            (1 - rollout_dones[:, 0]) * \
            ((gamma ** rollout_offsets[:, 0]) * rollout_nq)

        loss_rollout = F.smooth_l1_loss(q, expected_rollout_value.unsqueeze(1))

        max_q, argmax_q = policy_net(states, ens_num=ens_num).max(1)
        max_q = max_q.unsqueeze(1)
        argmax_q = argmax_q.unsqueeze(1)
        non_identity_value = ((1.-(actions == argmax_q).float()) * 0.01)
        loss_e = torch.sum(max_q + non_identity_value - q)

        total_loss += loss_q + lambda1 * loss_rollout + lambda2 * loss_e

    optimizer.zero_grad()
    total_loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return total_loss.detach()


def demo_optimization_ens_epochs(policy_net, target_net, memory, optimizer, gamma, lambda1, lambda2, batch_size, epochs, device):
    total_batch_size = epochs * batch_size
    bs, ba, br, bns, bd, brr, bro, brns, brd = memory.sample(total_batch_size)
    loss = 0
    for i in range(epochs):
        samples_bs = bs[i * batch_size:(i+1) * batch_size]
        samples_ba = ba[i * batch_size:(i+1) * batch_size]
        samples_br = br[i * batch_size:(i+1) * batch_size]
        samples_bns = bns[i * batch_size:(i+1) * batch_size]
        samples_bd = bd[i * batch_size:(i+1) * batch_size]
        samples_brr = brr[i * batch_size:(i+1) * batch_size]
        samples_bro = bro[i * batch_size:(i+1) * batch_size]
        samples_brns = brns[i * batch_size:(i+1) * batch_size]
        samples_brd = brd[i * batch_size:(i+1) * batch_size]

        samples = (samples_bs, samples_ba, samples_br, samples_bns,
                   samples_bd, samples_brr, samples_bro, samples_brns, samples_brd)

        loss += demo_optimization_ens_no_mem(policy_net, target_net,
                                             samples, optimizer, gamma, lambda1, lambda2, batch_size, device)
    return loss / epochs
