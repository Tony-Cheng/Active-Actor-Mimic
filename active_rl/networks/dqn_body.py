import torch.nn as nn
import torch.nn.functional as F
from ..utils import DiscreteActionConfig


class StandardDQNAtariBody(nn.Module):
    def __init__(self, config: DiscreteActionConfig):
        super(StandardDQNAtariBody, self).__init__()
        self.n_actions = config.n_actions
        self.input_channel = config.input_channel
        self.bn0 = nn.BatchNorm2d(self.input_channel)
        self.conv1 = nn.Conv2d(self.input_channel, 32,
                               kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    def forward(self, x):
        x = x.float()
        x = self.bn0(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)
