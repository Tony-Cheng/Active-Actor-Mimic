from collections import deque
from active_rl.utils.atari_utils import fp
import torch


def collect_trajectories(num_steps, env, action_selector, memory):
    q = deque(maxlen=5)
    done = True
    for step in range(num_steps):
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
