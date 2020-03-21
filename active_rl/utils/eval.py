from ..utils import BaseConfig


def evaluate(config: BaseConfig, num_episode=5):
    env = config.eval_env
    action_selector = config.action_selector
    e_rewards = []
    for _ in range(num_episode):
        env.reset()
        e_reward = 0
        done = False
        while not done:
            state = env.get_state()
            action = action_selector.select_action(state, False)
            next_state, reward, done, _ = env.step(action)
            e_reward += reward
        e_rewards.append(e_reward)

    return float(sum(e_rewards))/float(num_episode)
