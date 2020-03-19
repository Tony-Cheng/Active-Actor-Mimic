import torch.nn as nn
from ..utils import DiscreteActionConfig


class DDQNAgent(nn.Module):

    def __init__(self, config: DiscreteActionConfig, body):
        super(DDQNAgent, self).__init__()
        self.policy_net = body(config)
        self.target_net = body(config)
        self.target_net.load_state_dict(self.policy_net.state_dict)
        self.device = config.device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.memory = config.memory
        self.config = config

    def forward(self, state):
        return self.policy_net(state)

    def train(self):
        pass
