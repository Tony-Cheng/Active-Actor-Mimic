import torch


class BaseConfig:
    batch_size: int
    device: str

    def __init__(self):
        self.batch_size = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gamma = 0.99
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
        self.lr = None
        self.num_ensembles = None
        self.action_selection_policy = None


class StandardConfig:

    def __init__(self):
        self.device = None
        self.max_steps = 0


class DiscreteActionConfig(BaseConfig):

    def __init__(self):
        super(DiscreteActionConfig, self).__init__()
        self.eps_start = None
        self.eps_end = None
        self.eps_decay = None
        self.memory = None
        self.memory_size = None
        self.body = None
        self.optimizer = None
        self.agent = None
        self.train_freq = None
        self.target_update_freq = None
        self.height = None
        self.width = None
        self.input_channel = None
        self.memory_channel = None
        self.frame_channel = None
        self.n_actions = None


class AMNConfig(DiscreteActionConfig):

    def __init__(self):
        super(AMNConfig, self).__init__()
        self.max_steps = 0
        self.max_labels = 0
        self.agent = None
        self.expert_net_name = None
        self.eval_freq = None  # labels / eval step
        self.training_per_label = None
        self.perc_label = None
        self.steps_per_labeling = None
