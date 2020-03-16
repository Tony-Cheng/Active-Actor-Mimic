class StandardConfig:

    def __init__(self):
        self.device = None
        self.max_steps = 0


class AMNConfig:

    def __init__(self):
        self.device = None
        self.max_steps = 0
        self.max_labels = 0
        self.policy_net = None
        self.target_net = None
        self.discount = 0.99
        self.eval_freq = None  # labels / eval step
        self.training_per_label = None
        self.batch_size = None
        self.perc_label = None
        self.steps_per_labeling = None
