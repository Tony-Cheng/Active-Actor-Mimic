import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    A DQN network for playing atari games.
    """

    def __init__(self, in_channel=3, n_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc4 = nn.Linear(22 * 16 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)
