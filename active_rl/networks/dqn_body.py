import torch.nn as nn
import torch.nn.functional as F
from ..utils import DiscreteActionConfig


class StandardDQNAtariBody(nn.Module):
    def __init__(self, config: DiscreteActionConfig):
        super(StandardDQNAtariBody, self).__init__()
        self.n_actions = config.n_actions
        self.input_channel = config.input_channel
        self.conv1 = nn.Conv2d(self.input_channel, 32,
                               kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    def forward(self, x):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)
