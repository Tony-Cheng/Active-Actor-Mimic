from ..trainer import train_atari
from torch.utils.tensorboard import SummaryWriter
from ..networks import DDQNAgent, DQNAgent
from ..utils import DiscreteActionConfig, Replay, DiscreteActionSelector
from..environments import DiscreteAtariEnv


def build_ddqn_model(config: DiscreteActionConfig):
    config.env = DiscreteAtariEnv(config)
    f_channel, height, width, n_actions, n_channel = config.env.get_shape()
    config.n_actions = n_actions
    config.height = height
    config.width = width
    config.memory_channel = n_channel
    config.frame_channel = f_channel
    config.input_channel = n_channel - f_channel
    config.eval_env = DiscreteAtariEnv(config, eval=True)

    config.memory = Replay(config)

    config.agent = DDQNAgent(config)

    config.writer = SummaryWriter(f'runs/{config.writer_name}')
    config.action_selector = DiscreteActionSelector(config)

    train_atari(config)


def build_dqn_model(config: DiscreteActionConfig):
    config.env = DiscreteAtariEnv(config)
    f_channel, height, width, n_actions, n_channel = config.env.get_shape()
    config.n_actions = n_actions
    config.height = height
    config.width = width
    config.memory_channel = n_channel
    config.frame_channel = f_channel
    config.input_channel = n_channel - f_channel
    config.eval_env = DiscreteAtariEnv(config, eval=True)

    config.memory = Replay(config)

    config.agent = DQNAgent(config)

    config.writer = SummaryWriter(f'runs/{config.writer_name}')
    config.action_selector = DiscreteActionSelector(config)

    train_atari(config)
