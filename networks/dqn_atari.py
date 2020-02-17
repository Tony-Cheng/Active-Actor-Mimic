import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, n_actions)

        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.fc1.bias.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)

    def forward(self, x):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

# class DQN(nn.Module):
#     """
#     A DQN network for playing atari games.
#     """

#     def __init__(self, in_channel=3, n_actions=4):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel, 32, 8, stride=4)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.fc4 = nn.Linear(64 * 9 * 6, 512)
#         self.head = nn.Linear(512, n_actions)

#         nn.init.kaiming_uniform_(
#             self.conv1.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(
#             self.conv2.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(
#             self.conv3.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.constant_(self.conv1.bias, 0)
#         nn.init.constant_(self.conv2.bias, 0)
#         nn.init.constant_(self.conv3.bias, 0)

#     def forward(self, x, last_layer=False):
#         x = x.float() / 255
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.fc4(x.view(x.size(0), -1)))
#         if last_layer:
#             return self.head(x), x
#         else:
#             return self.head(x)


class MC_DQN(nn.Module):
    def __init__(self, in_channel=3, n_actions=4):
        super(MC_DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc4 = nn.Linear(64 * 9 * 6, 512)
        self.head = nn.Linear(512, n_actions)
        self.dropout = nn.Dropout(p=0.3)

        nn.init.kaiming_uniform_(
            self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(
            self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(
            self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)

    def forward(self, x, last_layer=False):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        if last_layer:
            return self.head(self.dropout(x)), x
        else:
            return self.head(self.dropout(x))


def to_policy(q_values, tau=0.1):
    return F.softmax(q_values / tau, dim=1)

