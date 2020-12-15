import torch
from batchbald_redux.batchbald import get_batchbald_batch, get_bald_batch
import random


class ReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, device):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, batch_size=None):
        if batch_size is None or len(self) < batch_size:
            batch_size = len(self)
        i = torch.randint(0, high=self.size, size=(batch_size,))
        bs = self.m_states[i, :4]
        bns = self.m_states[i, 1:]
        ba = self.m_actions[i]
        br = self.m_rewards[i].float()
        bd = self.m_dones[i].float()
        return bs, ba, br, bns, bd

    def __len__(self):
        return self.size


class _ReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, device):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, batch_size=None):
        if batch_size is None or len(self) < batch_size:
            batch_size = len(self)
        i = torch.randint(0, high=self.size, size=(batch_size,))
        bs = self.m_states[i]
        ba = self.m_actions[i]
        br = self.m_rewards[i].float()
        bd = self.m_dones[i].float()
        return bs, ba, br, bd

    def __len__(self):
        return self.size


class RankedReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, rank_func, AMN_net,
                 replacement=False, device='cuda'):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0
        self.rank_func = rank_func
        self.AMN_net = AMN_net
        self.replacement = replacement

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, percentage=0.1, batch_size=None):
        _, i = torch.sort(self.rank_func(
            self.AMN_net, self.m_states[: self.size, :4], device=self.device),
            descending=True)
        if batch_size is not None:
            i = i[: batch_size]
        else:
            i = i[: int(percentage * self.size)]
        i = i[torch.randperm(i.shape[0])]
        # i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i, :4]
        bns = self.m_states[i, 1:]
        ba = self.m_actions[i]
        br = self.m_rewards[i].float()
        bd = self.m_dones[i].float()
        if self.replacement:
            read_index = 0
            write_index = 0
            i_set = set(i.flatten())
            while read_index < self.size:
                if read_index in i_set:
                    read_index += 1
                else:
                    self.m_states[write_index] = self.m_states[read_index]
                    self.m_actions[write_index,
                                   0] = self.m_actions[read_index, 0]
                    self.m_rewards[write_index,
                                   0] = self.m_rewards[read_index, 0]
                    self.m_dones[write_index, 0] = self.m_dones[read_index, 0]
                    read_index += 1
                    write_index += 1
            self.size -= len(i_set)
            self.position -= len(i_set)
            self.position = self.position % self.size
        return bs, ba, br, bns, bd

    def __len__(self):
        return self.size


class GenericRankedReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, rank_func):
        c, h, w = state_shape
        self.capacity = capacity
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0
        self.rank_func = rank_func

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def label_percentage(self, percentage):
        _, i = torch.sort(self.rank_func(self.m_states[: self.size, :4]),
                          descending=True)
        i = i[: int(percentage * self.size)]
        i = i[torch.randperm(i.shape[0])]
        bs = self.m_states[i]
        ba = self.m_actions[i]
        br = self.m_rewards[i].float()
        bd = self.m_dones[i].float()
        return bs, ba, br, bd

    def __len__(self):
        return self.size


class GenericLabelledReplayMemory():
    def __init__(self, rank_buffer, labelled_buffer):
        self.labeled_buffer = labelled_buffer
        self.rank_buffer = rank_buffer

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.rank_buffer.push(state, action, reward, done)

    def label_sample_percentage(self, percentage):
        bs, ba, br, bd = self.rank_buffer.label_percentage(percentage)
        for i in range(bs.shape[0]):
            self.labeled_buffer.push(bs[i], ba[i, 0], br[i, 0], bd[i, 0])
        return bs.shape[0]

    def sample(self, batch_size):
        bs, ba, br, bns, bd = self.labeled_buffer.sample(batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labeled_buffer)


class _RankedReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, rank_func, AMN_net,
                 device='cuda'):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0
        self.rank_func = rank_func
        self.AMN_net = AMN_net

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, percentage=0.1):
        _, i = torch.sort(self.rank_func(
            self.AMN_net, self.m_states[: self.size, :4], device=self.device),
            descending=True)
        i = i[: int(percentage * self.size)]
        i = i[torch.randperm(i.shape[0])]
        # i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i]
        ba = self.m_actions[i]
        br = self.m_rewards[i].float()
        bd = self.m_dones[i].float()
        return bs, ba, br, bd

    def __len__(self):
        return self.size


class LabelledReplayMemory():
    def __init__(self, capacity_not_labelled, capacity_labelled, state_shape,
                 n_actions, rank_func, AMN_net, device='cuda'):
        self.device = device
        self.labeled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.rank_buffer = _RankedReplayMemory(
            capacity_not_labelled, state_shape, n_actions, rank_func, AMN_net,
            device=device)

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.rank_buffer.push(state, action, reward, done)

    def label_sample(self, percentage=0.1, batch_size=None):
        bs, ba, br, bd = self.rank_buffer.sample(
            percentage=percentage, batch_size=batch_size)
        for i in range(bs.shape[0]):
            self.labeled_buffer.push(bs[i], ba[i, 0], br[i, 0], bd[i, 0])
        return bs.shape[0]

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labeled_buffer.sample(batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labeled_buffer)


class _GeneralRankedReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, rank_func, AMN_net,
                 device='cuda'):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0
        self.rank_func = rank_func
        self.AMN_net = AMN_net

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, percentage=0.1):
        _, i = torch.sort(self.rank_func(
            self.AMN_net, self, device=self.device), descending=True)
        i = i[: int(percentage * self.size)]
        i = i[torch.randperm(i.shape[0])]
        # i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i]
        ba = self.m_actions[i]
        br = self.m_rewards[i].float()
        bd = self.m_dones[i].float()
        return bs, ba, br, bd

    def __len__(self):
        return self.size

    def get_all(self):
        bs = self.m_states[:, :4]
        bns = self.m_states[:, 1:]
        ba = self.m_actions
        br = self.m_rewards
        bd = self.m_dones.float()
        return bs, ba, br, bns, bd


class GeneralLabeledReplayMemory():
    def __init__(self, capacity_not_labelled, capacity_labelled, state_shape,
                 n_actions, rank_func, AMN_net, device='cuda'):
        self.device = device
        self.labeled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.rank_buffer = _GeneralRankedReplayMemory(
            capacity_not_labelled, state_shape, n_actions, rank_func, AMN_net,
            device=device)

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.rank_buffer.push(state, action, reward, done)

    def label_sample(self, percentage=0.1):
        bs, ba, br, bd = self.rank_buffer.sample(percentage=percentage)
        for i in range(bs.shape[0]):
            self.labeled_buffer.push(bs[i], ba[i, 0], br[i, 0], bd[i, 0])
        return bs.shape[0]

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labeled_buffer.sample(batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labeled_buffer)


class DuoMemory():
    def __init__(self, memory1, memory2, percentage=0.5):
        self.memory1 = memory1
        self.memory2 = memory2
        self.percentage = percentage

    def sample(self, batch_size=None):
        bs1, ba1, br1, bns1, bd1 = self.memory1.sample(
            batch_size=round(batch_size * self.percentage))
        bs2, ba2, br2, bns2, bd2 = self.memory2.sample(
            batch_size=round(batch_size * (1 - self.percentage)))
        bs = torch.cat((bs1, bs2))
        ba = torch.cat((ba1, ba2))
        br = torch.cat((br1, br2))
        bns = torch.cat((bns1, bns2))
        bd = torch.cat((bd1, bd2))
        return bs, ba, br, bns, bd

    def __len__(self):
        return min(len(self.memory1), len(self.memory2))


class _ObsRankedReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, rank_func, AMN_net,
                 device='cuda'):
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0
        self.rank_func = rank_func
        self.AMN_net = AMN_net

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state  # 5,84,84
        self.m_actions[self.position, 0] = action
        self.m_rewards[self.position, 0] = reward
        self.m_dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, percentage=0.1):
        obs = (self.m_states[: self.size, :4], self.m_actions[: self.size],
               self.m_rewards[: self.size], self.m_states[: self.size, 1:],
               self.m_dones[: self.size])
        _, i = torch.sort(self.rank_func(self.AMN_net, obs,
                                         device=self.device), descending=True)
        i = i[: int(percentage * self.size)]
        i = i[torch.randperm(i.shape[0])]
        # i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i]
        ba = self.m_actions[i]
        br = self.m_rewards[i].float()
        bd = self.m_dones[i].float()
        return bs, ba, br, bd

    def __len__(self):
        return self.size


class ObsLabeledReplayMemory():
    def __init__(self, capacity_not_labelled, capacity_labelled, state_shape,
                 n_actions, rank_func, AMN_net, device='cuda'):
        self.device = device
        self.labeled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.rank_buffer = _ObsRankedReplayMemory(
            capacity_not_labelled, state_shape, n_actions, rank_func, AMN_net,
            device=device)

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.rank_buffer.push(state, action, reward, done)

    def label_sample(self, percentage=0.1):
        bs, ba, br, bd = self.rank_buffer.sample(percentage=percentage)
        for i in range(bs.shape[0]):
            self.labeled_buffer.push(bs[i], ba[i, 0], br[i, 0], bd[i, 0])
        return bs.shape[0]

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labeled_buffer.sample(batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labeled_buffer)


class BatchBALDReplayMemoryForMcDropout():
    def __init__(self, capacity_not_labelled, capacity_labelled, batch_label_size, state_shape,
                 n_actions, AMN_net, num_samples, tau=0.1, device='cuda'):
        self.device = device
        self.n_actions = n_actions
        self.batch_label_size = batch_label_size
        self.num_samples = num_samples
        self.AMN_net = AMN_net
        self.unlabelled_buffer = _ReplayMemory(
            capacity_not_labelled, state_shape, n_actions, device)
        self.labelled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.tau = tau

    def push(self, state, action, reward, done):
        self.unlabelled_buffer.push(state, action, reward, done)

    def label_sample(self, batch_size=64, num_batch_samples=100):
        bs, ba, br, bd = self.unlabelled_buffer.sample()
        num_states = bs.shape[0]
        out = torch.empty(
            (num_states, self.num_samples, self.n_actions))
        for i in range(self.num_samples):
            for j in range(0, num_states, batch_size):
                next_batch_size = batch_size
                if (num_states - j < batch_size):
                    next_batch_size = num_states - j
                next_states = bs[j:j + next_batch_size, :4].to(self.device)
                with torch.no_grad():
                    out[j:j + next_batch_size, i,
                        :] = self.AMN_net(next_states)
        prob = (out / self.tau).softmax(dim=2)
        candidates = get_batchbald_batch(
            prob, self.batch_label_size, num_batch_samples, device=self.device)
        for indice in candidates.indices:
            self.labelled_buffer.push(
                bs[indice], ba[indice], br[indice], bd[indice])

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labelled_buffer.sample(
            batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labelled_buffer)


class BALDReplayMemoryForMcDropout():
    def __init__(self, capacity_not_labelled, capacity_labelled, batch_label_size, state_shape,
                 n_actions, AMN_net, num_samples, tau=0.1, device='cuda'):
        self.device = device
        self.n_actions = n_actions
        self.batch_label_size = batch_label_size
        self.num_samples = num_samples
        self.AMN_net = AMN_net
        self.unlabelled_buffer = _ReplayMemory(
            capacity_not_labelled, state_shape, n_actions, device)
        self.labelled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.tau = tau

    def push(self, state, action, reward, done):
        self.unlabelled_buffer.push(state, action, reward, done)

    def label_sample(self, batch_size=64):
        bs, ba, br, bd = self.unlabelled_buffer.sample()
        num_states = bs.shape[0]
        out = torch.empty(
            (num_states, self.num_samples, self.n_actions))
        for i in range(self.num_samples):
            for j in range(0, num_states, batch_size):
                next_batch_size = batch_size
                if (num_states - j < batch_size):
                    next_batch_size = num_states - j
                next_states = bs[j:j + next_batch_size, :4].to(self.device)
                with torch.no_grad():
                    out[j:j + next_batch_size, i] = self.AMN_net(next_states)
        prob = (out / self.tau).softmax(dim=2)

        candidates = get_bald_batch(
            prob, self.batch_label_size, device=self.device)

        for indice in candidates.indices:
            self.labelled_buffer.push(
                bs[indice], ba[indice], br[indice], bd[indice])

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labelled_buffer.sample(
            batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labelled_buffer)


class RandomReplayMemoryForMcDropout():
    def __init__(self, capacity_not_labelled, capacity_labelled, batch_label_size, state_shape,
                 n_actions, AMN_net, num_samples, device='cuda'):
        self.device = device
        self.n_actions = n_actions
        self.batch_label_size = batch_label_size
        self.num_samples = num_samples
        self.AMN_net = AMN_net
        self.unlabelled_buffer = _ReplayMemory(
            capacity_not_labelled, state_shape, n_actions, device)
        self.labelled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)

    def push(self, state, action, reward, done):
        self.unlabelled_buffer.push(state, action, reward, done)

    def label_sample(self, batch_size=64):
        bs, ba, br, bd = self.unlabelled_buffer.sample()
        num_states = bs.shape[0]
        indices = random.sample(range(num_states), self.batch_label_size)
        for indice in indices:
            self.labelled_buffer.push(
                bs[indice], ba[indice], br[indice], bd[indice])

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labelled_buffer.sample(
            batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labelled_buffer)


class BALDReplayMemoryForEnsDQN():
    def __init__(self, capacity_not_labelled, capacity_labelled, batch_label_size, state_shape,
                 n_actions, AMN_net, tau=0.1, device='cuda'):
        self.device = device
        self.n_actions = n_actions
        self.batch_label_size = batch_label_size
        self.AMN_net = AMN_net
        self.unlabelled_buffer = _ReplayMemory(
            capacity_not_labelled, state_shape, n_actions, device)
        self.labelled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.tau = tau

    def push(self, state, action, reward, done):
        self.unlabelled_buffer.push(state, action, reward, done)

    def label_sample(self, batch_size=64):
        bs, ba, br, bd = self.unlabelled_buffer.sample()
        num_states = bs.shape[0]
        out = torch.empty(
            (num_states, self.AMN_net.get_num_ensembles(), self.n_actions))
        for i in range(self.AMN_net.get_num_ensembles()):
            for j in range(0, num_states, batch_size):
                next_batch_size = batch_size
                if (num_states - j < batch_size):
                    next_batch_size = num_states - j
                next_states = bs[j:j + next_batch_size, :4].to(self.device)
                with torch.no_grad():
                    out[j:j + next_batch_size,
                        i] = self.AMN_net(next_states, ens_num=i)
        prob = (out / self.tau).softmax(dim=2)

        candidates = get_bald_batch(
            prob, self.batch_label_size, device=self.device)

        for indice in candidates.indices:
            self.labelled_buffer.push(
                bs[indice], ba[indice], br[indice], bd[indice])

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labelled_buffer.sample(
            batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labelled_buffer)


class BatchBALDReplayMemoryForEnsDQN():
    def __init__(self, capacity_not_labelled, capacity_labelled, batch_label_size, state_shape,
                 n_actions, AMN_net, tau=0.1, device='cuda'):
        self.device = device
        self.n_actions = n_actions
        self.batch_label_size = batch_label_size
        self.AMN_net = AMN_net
        self.unlabelled_buffer = _ReplayMemory(
            capacity_not_labelled, state_shape, n_actions, device)
        self.labelled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.tau = tau

    def push(self, state, action, reward, done):
        self.unlabelled_buffer.push(state, action, reward, done)

    def label_sample(self, batch_size=64, num_batch_samples=100):
        bs, ba, br, bd = self.unlabelled_buffer.sample()
        num_states = bs.shape[0]
        out = torch.empty(
            (num_states, self.AMN_net.get_num_ensembles(), self.n_actions))
        for i in range(self.AMN_net.get_num_ensembles()):
            for j in range(0, num_states, batch_size):
                next_batch_size = batch_size
                if (num_states - j < batch_size):
                    next_batch_size = num_states - j
                next_states = bs[j:j + next_batch_size, :4].to(self.device)
                with torch.no_grad():
                    out[j:j + next_batch_size,
                        i] = self.AMN_net(next_states, ens_num=i)
        prob = (out / self.tau).softmax(dim=2)

        candidates = get_batchbald_batch(
            prob, self.batch_label_size, num_batch_samples, device=self.device)

        for indice in candidates.indices:
            self.labelled_buffer.push(
                bs[indice], ba[indice], br[indice], bd[indice])

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labelled_buffer.sample(
            batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labelled_buffer)


class DoubleRankedReplayMemoryForEnsDQN():
    def __init__(self, capacity_not_labelled, capacity_labelled, batch_label_size, state_shape,
                 n_actions, net1, net2, rank_func1, rank_func2, tau=0.1, device='cuda'):
        self.device = device
        self.n_actions = n_actions
        self.batch_label_size = batch_label_size
        self.net1 = net1
        self.net2 = net2
        self.rank_func1 = rank_func1
        self.rank_func2 = rank_func2
        self.unlabelled_buffer = _ReplayMemory(
            capacity_not_labelled, state_shape, n_actions, device)
        self.labelled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.tau = tau

    def push(self, state, action, reward, done):
        self.unlabelled_buffer.push(state, action, reward, done)

    def label_sample(self, batch_size=64, device='cuda'):
        bs, ba, br, bd = self.unlabelled_buffer.sample()
        rank_val1 = self.rank_func1(
            self.net1, bs[:, :4], batch_size=batch_size, device=device)
        rank_val2 = self.rank_func2(
            self.net2, bs[:, :4], batch_size=batch_size, device=device)

        rank_val1 += torch.min(rank_val1)
        rank_val1 /= torch.max(rank_val1)

        rank_val2 += torch.min(rank_val2) + 0.5
        rank_val2 /= torch.max(rank_val2)

        rank_val = rank_val1 * rank_val2

        _, i = torch.sort(rank_val, descending=True)

        i = i[: self.batch_label_size]

        i = i[torch.randperm(i.shape[0])]

        for j in range(self.batch_label_size):
            next_index = i[j]
            self.labelled_buffer.push(
                bs[next_index], ba[next_index], br[next_index], bd[next_index])

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labelled_buffer.sample(
            batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labelled_buffer)


class DoubleRankedReplayMemoryForEnsDQNV2():
    def __init__(self, capacity_not_labelled, capacity_labelled, batch_label_size, state_shape,
                 n_actions, net1, net2, rank_func1, rank_func2, tau=0.1, device='cuda'):
        self.device = device
        self.n_actions = n_actions
        self.batch_label_size = batch_label_size
        self.net1 = net1
        self.net2 = net2
        self.rank_func1 = rank_func1
        self.rank_func2 = rank_func2
        self.unlabelled_buffer = _ReplayMemory(
            capacity_not_labelled, state_shape, n_actions, device)
        self.labelled_buffer = ReplayMemory(
            capacity_labelled, state_shape, n_actions, device)
        self.tau = tau

    def push(self, state, action, reward, done):
        self.unlabelled_buffer.push(state, action, reward, done)

    def label_sample(self, batch_size=64, device='cuda'):
        bs, ba, br, bd = self.unlabelled_buffer.sample()
        rank_val1 = self.rank_func1(
            self.net1, bs, batch_size=batch_size, device=device)
        rank_val2 = self.rank_func2(
            self.net2, bs, batch_size=batch_size, device=device)

        rank_val1 += torch.min(rank_val1)
        rank_val1 /= torch.max(rank_val1)

        rank_val2 += torch.min(rank_val2) + 0.5
        rank_val2 /= torch.max(rank_val2)

        rank_val = rank_val1 * rank_val2

        _, i = torch.sort(rank_val, descending=True)

        i = i[: self.batch_label_size]

        i = i[torch.randperm(i.shape[0])]

        for j in range(self.batch_label_size):
            next_index = i[j]
            self.labelled_buffer.push(
                bs[next_index], ba[next_index], br[next_index], bd[next_index])

    def sample(self, batch_size=None):
        bs, ba, br, bns, bd = self.labelled_buffer.sample(
            batch_size=batch_size)
        return bs, ba, br, bns, bd

    def __len__(self):
        return len(self.labelled_buffer)
