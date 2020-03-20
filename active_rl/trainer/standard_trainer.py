from ..utils import DiscreteActionConfig, evaluate
from tqdm.notebook import tqdm


def train_atari(config: DiscreteActionConfig):
    agent = config.agent
    memory = config.memory
    initial_steps = config.initial_steps
    writer = config.writer
    env = config.env
    max_steps = config.max_steps
    action_selector = config.action_selector
    train_freq = config.train_freq
    eval_freq = config.eval_freq
    save_freq = config.save_freq
    target_update_freq = config.target_update_freq
    save_filename = config.save_filename
    progressive = tqdm(range(max_steps), total=max_steps,
                       ncols=400, leave=False, unit='b')
    done = False
    env.reset()
    for step in progressive:
        state = env.get_state()
        action = action_selector.select_action(state)
        _, reward, done, info = env.step(action)
        all_states = env.get_all_states()
        memory.push(all_states, action, reward, done)

        if step > initial_steps:
            if step % train_freq == 0:
                loss = agent.train()
                if writer is not None:
                    writer.add_scalar('loss_step', loss, step)

            if step % eval_freq == 0:
                rewards = evaluate(config)
                if writer is not None:
                    writer.add_scalar('reward_step', rewards, step)

            if step % target_update_freq == 0:
                agent.update_target()

            if step % save_freq == 0:
                agent.save(save_filename)

        if done:
            env.reset()
