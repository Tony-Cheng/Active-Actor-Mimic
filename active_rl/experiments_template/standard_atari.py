from ..trainer import train_atari
from torch.utils.tensorboard import SummaryWriter
from ..networks import DDQNAgent
from ..utils import DiscreteActionConfig, Replay, DiscreteActionSelector
from..environments import DiscreteAtariEnv


def build_ddqn_model(config: DiscreteActionConfig):
    config.env = DiscreteAtariEnv(config)
    config.n_actions = config.env.get_n_actions()
    config.eval_env = DiscreteAtariEnv(config, eval=True)
    config.agent = DDQNAgent(config)
    config.memory = Replay(config)
    config.input_channel = 5
    config.writer = SummaryWriter(config.writer_name)
    config.action_selector = DiscreteActionSelector(config)
    train_atari(config)
