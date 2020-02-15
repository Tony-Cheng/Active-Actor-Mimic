import torch
from collections import namedtuple
import torch.nn.functional as F

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


def standard_optimization(policy_net, target_net, memory, optimizer, BATCH_SIZE=128, GAMMA=0.99, device='cuda'):
    """
    Apply the standard procedure to deep Q network.
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([[r]], device=device), batch.reward)))

    non_final_mask = torch.tensor(tuple(
        map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]).to('cuda')

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp(-1, 1)
    optimizer.step()

    return loss
