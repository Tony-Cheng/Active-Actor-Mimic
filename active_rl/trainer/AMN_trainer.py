from collections import deque
from tqdm.notebook import tqdm
from ..utils import fp, evaluate
from ..environments import wrap_deepmind
import torch


def train_AMN(policy_net, expert_net, optimizer, memory, action_selector,
              train_iters, train_func, env_raw, num_steps, initial_steps,
              label_perc, not_labelled_capacity, batch_size=128, writer=None,
              device='cuda'):
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=False,
                        clip_rewards=True)
    frame_queue = deque(maxlen=5)
    done = True
    progressive = tqdm(range(num_steps), total=num_steps,
                       ncols=400, leave=False, unit='b')
    num_labels = 0
    for step in progressive:
        if done:
            env.reset()
            img, _, _, _ = env.step(1)
            for i in range(10):
                n_frame, _, _, _ = env.step(0)
                n_frame = fp(n_frame)
                frame_queue.append(n_frame)

        state = torch.cat(list(frame_queue))[1:].unsqueeze(0)
        action, eps = action_selector.select_action(state)
        n_frame, reward, done, info = env.step(action)
        n_frame = fp(n_frame)

        frame_queue.append(n_frame)
        memory.push(torch.cat(list(frame_queue)).unsqueeze(0), action, reward,
                    done)

        if step > initial_steps and step % not_labelled_capacity == 0:
            labels = memory.label_sample(percentage=label_perc)
            num_labels += labels
            loss = 0
            for _ in range(train_iters):
                loss += train_func(policy_net, expert_net, optimizer, memory,
                                   batch_size=batch_size, device=device)
            if writer is not None:
                writer.add_scalar("loss_label", loss, num_labels)
                evaluated_reward = evaluate(policy_net, env_raw,
                                            action_selector)
                writer.add_scalar("reward_label", evaluated_reward, num_labels)
