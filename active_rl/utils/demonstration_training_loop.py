from tqdm.notebook import tqdm
from active_rl.utils.demonstration_memory import RolloutOfflineReplayMemory, GenericRankedRolloutOfflineReplayMemory
from active_rl.utils.optimization import standard_optimization_ensemble
from active_rl.utils.demonstration_optimization import standard_optimization, demo_optimization_ens_epochs
from active_rl.utils.atari_utils import evaluate
import random
from collections import deque
from active_rl.utils.atari_utils import fp, ActionSelector
import torch
from active_rl.utils.memory import ReplayMemory


standard_config = {
    'num_steps_per_memory': 100000,
    'num_steps_per_eval': 10000,
    'num_steps_per_target_net_update': 4000,
    'num_memory': 1700,
    'file_name': '',
    'gamma': 0.99,
    'rollout_length': 40,
    'lambda1': 0.2,
    'lambda2': 0.05,
    'policy_net': None,
    'target_net': None,
    'optimizer': None,
    'writer': None,
    'env_raw': None,
    'n_actions': 0,
    'num_episodes': 15,
    'batch_size': 64,
    'device': None,
    'rank_func': None,
    'epochs': 40
}

std_demo_config = {
    'num_steps': 20000000,
    'env': None,
    'env_raw': None,
    'memory_capacity': 400000,
    'device': None,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'eps_decay': 1000000,
    'policy_net': None,
    'target_net': None,
    'optimizer': None,
    'batch_size': 64,
    'writer': None,
    'policy_update': 4,
    'target_update': 4000,
    'eval_freq': 10000,
    'eval_num_episodes': 15,
    'demo_config': None,
    'demo_training_freq': 5000,
    'std_training_func': standard_optimization_ensemble
}


def standard_training_loop(config):
    num_memory = config['num_memory']
    num_steps_per_memory = config['num_steps_per_memory']
    progressive = tqdm(range(num_memory), total=num_memory,
                       ncols=400, leave=False, unit='b')
    policy_net = config['policy_net']
    target_net = config['target_net']
    optimizer = config['optimizer']
    device = config['device']
    batch_size = config['batch_size']
    lambda1 = config['lambda1']
    lambda2 = config['lambda2']
    gamma = config['gamma']
    writer = config['writer']
    num_steps_per_eval = config['num_steps_per_eval']
    num_steps_per_target_net_update = config['num_steps_per_target_net_update']
    env_raw = config['env_raw']
    n_actions = config['n_actions']
    num_episodes = config['num_episodes']
    step = 0
    for memory_id in progressive:
        memory = RolloutOfflineReplayMemory(
            config['file_name'] + f'/{memory_id}.pkl', config['gamma'], config['rollout_length'])
        progressive_training_steps = tqdm(range(
            num_steps_per_memory), total=num_steps_per_memory, ncols=400, leave=False, unit='b')
        for training_step in progressive_training_steps:
            step += 1
            standard_optimization(
                policy_net, target_net, memory, optimizer, gamma, lambda1, lambda2, batch_size, device)
            if step > 0 and step % num_steps_per_eval == 0:
                evaluated_reward = evaluate(
                    None, policy_net, device, env_raw, n_actions, eps=0.05, num_episode=num_episodes)
                writer.add_scalar('rewards', evaluated_reward,
                                  memory_id * num_steps_per_memory + training_step)
            if step > 0 and step % num_steps_per_target_net_update == 0:
                target_net.load_state_dict(policy_net.state_dict())


def active_demonstration_training_single_loop(config):
    num_memory = config['num_memory']
    policy_net = config['policy_net']
    target_net = config['target_net']
    optimizer = config['optimizer']
    device = config['device']
    batch_size = config['batch_size']
    lambda1 = config['lambda1']
    lambda2 = config['lambda2']
    gamma = config['gamma']
    epochs = config['epochs']
    memory_id = random.randint(0, num_memory - 1)
    memory = GenericRankedRolloutOfflineReplayMemory(
        config['file_name'] + f'/{memory_id}.pkl', config['gamma'], config['rollout_length'], config['rank_func'])
    return demo_optimization_ens_epochs(policy_net, target_net, memory,
                                        optimizer, gamma, lambda1, lambda2, batch_size, epochs, device)


def active_demo_std_training(config):
    num_steps = config['num_steps']
    env = config['env']
    env_raw = config['env_raw']
    device = config['device']
    eps_start = config['eps_start']
    eps_end = config['eps_end']
    eps_decay = config['eps_decay']
    policy_net = config['policy_net']
    target_net = config['target_net']
    optimizer = config['optimizer']
    writer = config['writer']
    policy_update = config['policy_update']
    target_update = config['target_update']
    eval_freq = config['eval_freq']
    batch_size = config['batch_size']
    eval_num_episodes = config['eval_num_episodes']
    demo_training_freq = config['demo_training_freq']
    demo_config = config['demo_config']
    memory_capacity = config['memory_capacity']
    std_training_func = config['std_training_func']
    n_actions = env.action_space.n
    c, h, w = fp(env.reset()).shape
    memory = ReplayMemory(memory_capacity, [5, h, w], n_actions, device)
    action_selector = ActionSelector(
        eps_start, eps_end, policy_net, eps_decay, n_actions, device)
    progressive = tqdm(
        range(num_steps), total=num_steps, ncols=400, leave=False, unit='b')
    done = True
    q = deque(maxlen=5)
    for step in progressive:
        if done:
            env.reset()
            img, _, _, _ = env.step(1)  # BREAKOUT specific !!!
            for i in range(10):  # no-op
                n_frame, _, _, _ = env.step(0)
                n_frame = fp(n_frame)
                q.append(n_frame)

        # Select and perform an action
        state = torch.cat(list(q))[1:].unsqueeze(0)
        action, eps = action_selector.select_action(state)
        n_frame, reward, done, info = env.step(action)
        n_frame = fp(n_frame)

        # 5 frame as memory
        q.append(n_frame)
        # here the n_frame means next frame from the previous time step
        memory.push(torch.cat(list(q)).unsqueeze(0), action, reward, done)

        if step > 0 and step % policy_update == 0:
            loss = std_training_func(
                policy_net, target_net, optimizer, memory, batch_size=batch_size, device=device)
            writer.add_scalar('std_dqn_ens_loss', loss, step)

        if step > 0 and step % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step > 0 and step % eval_freq == 0:
            evaluated_reward = evaluate(
                None, policy_net, device, env_raw, n_actions, eps=0.05, num_episode=eval_num_episodes)
            writer.add_scalar('rewards', evaluated_reward, step)

        if step > 0 and demo_training_freq != 0 and step % demo_training_freq == 0:
            loss = active_demonstration_training_single_loop(demo_config)
            writer.add_scalar('demo_dqn_ens_loss', loss, step)
