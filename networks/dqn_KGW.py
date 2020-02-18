import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False, padding=1)
        self.fc1 = nn.Linear(64*9*9, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)