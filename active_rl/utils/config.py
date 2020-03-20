import torch
from ..networks import BaseAgent
from ..environments import EnvInterface


class BaseConfig:
    batch_size: int
    device: str
    env: EnvInterface

    def __init__(self):
        self.batch_size = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.discount = 0.99
        self.env_name = None
        self.env = None
        self.eval_env = None
        self.intial_steps = None
        self.max_steps = None
        self.save_freq = None
        self.eval_freq = None
        self.writer = None
        self.writer_name = None
        self.action_selector = None
        self.save_filename = None


class StandardConfig:

    def __init__(self):
        self.device = None
        self.max_steps = 0


class DiscreteActionConfig(BaseConfig):
    agent: BaseAgent

    def __init__(self):
        super(DiscreteActionConfig, self).__init__()
        self.eps_start = None
        self.eps_end = None
        self.eps_decay = None
        self.input_channel = None
        self.memory = None
        self.memory_size = None
        self.body = None
        self.optimizer = None
        self.save_freq = None
        self.agent = None
        self.train_freq = None
        self.target_update_freq = None
        self.height = None
        self.width = None
        self.n_actions = None


class AMNConfig(DiscreteActionConfig):

    def __init__(self):
        super(AMNConfig, self).__init__()
        self.max_steps = 0
        self.max_labels = 0
        self.policy_net = None
        self.target_net = None
        self.eval_freq = None  # labels / eval step
        self.training_per_label = None
        self.perc_label = None
        self.steps_per_labeling = None
