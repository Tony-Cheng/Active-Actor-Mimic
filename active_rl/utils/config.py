import torch


class BaseConfig:
    batch_size: int
    device: str

    def __init__(self):
        self.batch_size = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.discount = 0.99


class StandardConfig:

    def __init__(self):
        self.device = None
        self.max_steps = 0


class AMNConfig(BaseConfig):

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
