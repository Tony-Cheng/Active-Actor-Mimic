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

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x, last_layer=False):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        if last_layer:
            return self.fc2(x), x
        else:
            return self.fc2(x)


class DVN(nn.Module):
    def __init__(self):
        super(DVN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, 1)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x, last_layer=False):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        if last_layer:
            return self.fc2(x), x
        else:
            return self.fc2(x)


class DQN_Large(nn.Module):
    def __init__(self, n_actions):
        super(DQN_Large, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128*7*7, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, n_actions)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x, last_layer=False):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MC_DQN(nn.Module):
    def __init__(self, in_channel=3, n_actions=4, dropout_p=0.1):
        super(MC_DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.dropout = nn.Dropout(p=dropout_p)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x, last_layer=False):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        if last_layer:
            return self.fc2(self.dropout(x)), x
        else:
            return self.fc2(self.dropout(x))


class ENS_DQN(nn.Module):
    def __init__(self, n_actions=4, num_networks=5):
        super(ENS_DQN, self).__init__()
        self.ensembles = []
        self.num_networks = num_networks

        for _ in range(num_networks):
            new_net = DQN(n_actions)
            self.ensembles.append(new_net)

    def init_weights(self, m):
        if type(m) == list:
            for net in m:
                net.apply(net.init_weights)

    def to(self, device):
        for net in self.ensembles:
            net.to(device)
        return self

    def parameters(self):
        params = []
        for net in self.ensembles:
            params += net.parameters()
        return params

    def forward(self, x, tau=0.1, ens_num=None, last_layer=False,
                q_value=False):
        if ens_num is None:
            prediction = 0
            for net in self.ensembles:
                prediction += to_policy(net(x), tau=tau)
            return prediction / len(self.ensembles)
        elif q_value:
            prediction = 0
            for net in self.ensembles:
                prediction += (net(x))
            return prediction / len(self.ensembles)
        else:
            net = self.ensembles[ens_num]
            return net(x, last_layer=last_layer)

    def get_num_ensembles(self):
        return len(self.ensembles)

    def state_dict(self):
        dicts = []
        for i in range(len(self.ensembles)):
            dicts.append(self.ensembles[i].state_dict())
        return dicts

    def load_state_dict(self, dicts):
        for i in range(len(dicts)):
            self.ensembles[i].load_state_dict(dicts[i])

    def eval(self):
        for i in range(len(self.ensembles)):
            self.ensembles[i].eval()


class ENS_DQN_Large(ENS_DQN):
    def __init__(self, n_actions=4, num_networks=5):
        super(ENS_DQN_Large, self).__init__()
        self.ensembles = []
        self.num_networks = num_networks

        for _ in range(num_networks):
            new_net = DQN_Large(n_actions)
            self.ensembles.append(new_net)


class Bootstrap_DQN(nn.Module):
    def __init__(self, n_actions=4, num_networks=5):
        super(Bootstrap_DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.bootstrap = nn.ModuleList(
            [nn.Linear(512, n_actions) for i in range(num_networks)])
        self.num_networks = num_networks

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x, training=False):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))

        if training:
            return [self.bootstrap[i](x) for i in range(len(self.bootstrap))]
        else:
            q_vals = 0
            for i in range(len(self.bootstrap)):
                q_vals += self.bootstrap[i](x)
            return q_vals / len(self.bootstrap)


class SmallSharingBootstrap_DQN(nn.Module):
    def __init__(self, n_actions=4, num_networks=5):
        super(SmallSharingBootstrap_DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
             for i in range(num_networks)])
        self.conv3 = nn.ModuleList(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
             for i in range(num_networks)])
        self.fc1 = nn.ModuleList(
            [nn.Linear(64*7*7, 512) for i in range(num_networks)])
        self.bootstrap = nn.ModuleList(
            [nn.Linear(512, n_actions) for i in range(num_networks)])
        self.num_networks = num_networks

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x, training=False):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = [F.relu(self.conv2[i](x)) for i in range(len(self.bootstrap))]
        x = [F.relu(self.conv3[i](x[i])) for i in range(len(self.bootstrap))]
        x = [F.relu(self.fc1[i](x[i].view(x[i].size(0), -1)))
             for i in range(len(self.bootstrap))]

        if training:
            return [self.bootstrap[i](x[i]) for i in range(len(self.bootstrap))]
        else:
            q_vals = 0
            for i in range(len(self.bootstrap)):
                q_vals += self.bootstrap[i](x[i])
            return q_vals / len(self.bootstrap)


class IndependentBootstrap_DQN(nn.Module):
    def __init__(self, n_actions=4, num_networks=5):
        super(IndependentBootstrap_DQN, self).__init__()
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
             for i in range(num_networks)])
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
             for i in range(num_networks)])
        self.conv3 = nn.ModuleList(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
             for i in range(num_networks)])
        self.fc1 = nn.ModuleList(
            [nn.Linear(64*7*7, 512) for i in range(num_networks)])
        self.bootstrap = nn.ModuleList(
            [nn.Linear(512, n_actions) for i in range(num_networks)])
        self.num_networks = num_networks

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x, training=False):
        x = x.float() / 255.
        x = [F.relu(self.conv1[i](x)) for i in range(len(self.bootstrap))]
        x = [F.relu(self.conv2[i](x[i])) for i in range(len(self.bootstrap))]
        x = [F.relu(self.conv3[i](x[i])) for i in range(len(self.bootstrap))]
        x = [F.relu(self.fc1[i](x[i].view(x[i].size(0), -1)))
             for i in range(len(self.bootstrap))]

        if training:
            return [self.bootstrap[i](x[i]) for i in range(len(self.bootstrap))]
        else:
            q_vals = 0
            for i in range(len(self.bootstrap)):
                q_vals += self.bootstrap[i](x[i])
            return q_vals / len(self.bootstrap)


class ENS_DQN_V2(nn.Module):
    def __init__(self, n_actions=4, num_networks=5):
        super(ENS_DQN_V2, self).__init__()
        self.ensembles = []
        self.num_networks = num_networks

        for _ in range(num_networks):
            new_net = DQN(n_actions)
            self.ensembles.append(new_net)

    def init_weights(self, m):
        if type(m) == list:
            for net in m:
                net.apply(net.init_weights)

    def to(self, device):
        for i in range(len(self.ensembles)):
            self.ensembles[i].to(device)
        return self

    def parameters(self):
        params = []
        for i in range(len(self.ensembles)):
            params += self.ensembles[i].parameters()
        return params

    def forward(self, x, tau=0.1, ens_num=None, last_layer=False,
                q_value=False):
        if ens_num is None:
            prediction = 0
            for i in range(len(self.ensembles)):
                prediction += to_policy(self.ensembles[i](x), tau=tau)
            return prediction / len(self.ensembles)
        elif q_value:
            prediction = 0
            for i in range(len(self.ensembles)):
                prediction += (self.ensembles[i](x))
            return prediction / len(self.ensembles)
        else:
            net = self.ensembles[ens_num]
            return net(x, last_layer=last_layer)

    def get_num_ensembles(self):
        return len(self.ensembles)

    def state_dict(self):
        dicts = []
        for i in range(len(self.ensembles)):
            dicts.append(self.ensembles[i].state_dict())
        return dicts

    def load_state_dict(self, dicts):
        for i in range(len(dicts)):
            self.ensembles[i].load_state_dict(dicts[i])

    def eval(self):
        for i in range(len(self.ensembles)):
            self.ensembles[i].eval()


def to_policy(q_values, tau=0.1):
    return F.softmax(q_values / tau, dim=1)
