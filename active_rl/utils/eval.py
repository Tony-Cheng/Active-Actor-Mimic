from ..utils import BaseConfig


def evaluate(config: BaseConfig, num_episode=5):
    env = config.eval_env
    action_selector = config.action_selector
    e_rewards = []
    done = False
    for _ in range(num_episode):
        state = env.reset()
        e_reward = 0
        while not done:
            action = action_selector.select_action(state, False)
            next_state, reward, done, _ = env.step(action)
            e_reward += reward
        e_rewards.append(e_reward)

    return float(sum(e_rewards))/float(num_episode)
