import torch


def hypothesis_collective_actions(model, memory, batch_size, device='cuda'):
    states, actions, _, _, _ = memory.sample(batch_size)
    actions = actions.squeeze()
    with torch.no_grad():
        model_actions = model(states.to(device)).max(1)[1].cpu()
    num_same_actions = torch.sum(actions == model_actions)
    return num_same_actions


def hypothesis_avg_member_actions(model, memory, batch_size, device='cuda'):
    states, actions, _, _, _ = memory.sample(batch_size)
    actions = actions.squeeze()
    num_same_actions = 0
    for ens_num in range(model.get_num_ensembles()):
        with torch.no_grad():
            model_actions = model(
                states.to(device), ens_num=ens_num).max(1)[1].cpu()
        num_same_actions += torch.sum(actions == model_actions)
    return 1.0 * num_same_actions / model.get_num_ensembles()
