import torch


class BaseConfig:
    batch_size: int
    device: str

    def __init__(self):
        self.batch_size = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.discount = 0.99
        self.env_name = None
        self.env = None


class StandardConfig:

    def __init__(self):
        self.device = None
        self.max_steps = 0


class DiscreteActionConfig(BaseConfig):

    def __init__(self):
        self.eps_start = None
        self.eps_end = None
        self.policy_net = None
        self.eps_decay = None
        self.eps_start = None
        self.eps_end = None
        self.eps_decay = None
        self.input_channel = None
        self.memory = None
        self.memory_size = None


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
