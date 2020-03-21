from collections import deque
from tqdm.notebook import tqdm
from ..utils import fp, evaluate
from ..environments import wrap_deepmind
import torch


def train_standard(policy_net, target_net, optimizer, memory, action_selector,
                   train_freq, target_update_freq, train_func, env_raw,
                   num_steps, initial_steps, eval_freq, batch_size=128,
                   filename=None, save_freq=None, writer=None, device='cuda'):
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=False,
                        clip_rewards=True)
    frame_queue = deque(maxlen=5)
    done = True
    progressive = tqdm(range(num_steps), total=num_steps,
                       ncols=400, leave=False, unit='b')
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

        if step > initial_steps:
            if step % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if step % train_freq == 0:
                loss = train_func(policy_net, target_net, optimizer,
                                  memory, batch_size=batch_size, device=device)
                if writer is not None:
                    writer.add_scalar("loss", loss, step)
                    writer.add_scalar("epsilon", eps, step)
            if step % eval_freq == 0:
                evaluated_reward = evaluate(
                    policy_net, env_raw, action_selector)
                writer.add_scalar("reward", evaluated_reward, step)

            if save_freq is not None and step % save_freq == 0:
                torch.save(policy_net, filename)
