import torch
from collections import namedtuple
import torch.nn.functional as F


def standard_optimization(policy_net, target_net, optimizer, memory, batch_size=128, GAMMA=0.99, training='True', device='cuda'):
    if not training:
        return None
    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(batch_size)

    q = policy_net(state_batch).gather(1, action_batch)
    nq = target_net(n_state_batch).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (nq * GAMMA)*(1.-done_batch[:,0]) + reward_batch[:,0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.detach()


# def AMN_optimization(AMN_net, expert_net, memory, optimizer, feature_regression=False, beta=0.01, BATCH_SIZE=256, GAMMA=0.99, device='cuda'):
#     """
#     Apply the standard procedure to deep Q network.
#     """
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     batch = Transition(*zip(*transitions))

#     state_batch = torch.cat(batch.state).to(device)
#     if feature_regression == True:
#         AMN_q_value, AMN_last_layer = AMN_net(state_batch, last_layer=True)
#         expert_q_value, expert_last_layer = expert_net(state_batch, last_layer=True)
#         loss = F.mse_loss(AMN_last_layer, expert_last_layer)
#     else:
#         AMN_q_value = AMN_net(state_batch)
#         expert_q_value = expert_net(state_batch)
#         loss = 0

#     AMN_policy = to_policy(AMN_q_value)
#     expert_policy = to_policy(expert_q_value).detach()

#     loss += torch.sum(expert_policy * torch.log(AMN_policy))

#     optimizer.zero_grad()
#     loss.backward()
#     for param in AMN_net.parameters():
#         param.grad.data.clamp(-1, 1)
#     optimizer.step()

#     return loss


def to_policy(q_values, tau=0.1):
    return F.softmax(q_values / tau, dim=1)
