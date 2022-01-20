import torch
import math
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity, gamma, device):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.n_multi_step = 3
        self.gamma = gamma

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def nstep_sample(self, batch_size):
        """calculate n-step reward"""
        batch = []
        for i in range(batch_size):
            finish = random.randint(self.n_multi_step, len(self.memory))
            begin = finish-self.n_multi_step
            sum_reward = 0  # n_step rewards
            data = self.memory[begin:finish]
            state = data[0].state
            action = data[0].action
            for j in range(self.n_multi_step):
                # compute the n-th reward
                sum_reward += (self.gamma**j) * data[j].reward
                if not data[j].mask:
                    # manage end of episode
                    next_state = data[j].state
                    mask = data[j].mask
                    break
                else:
                    next_state = data[j].state
                    mask = data[j].mask

            batch.append(Transition(
                state, action, mask, next_state, sum_reward))
        return batch

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory:
    def __init__(self, capacity, gamma, device):
        """hyperparameters"""
        self.alpha = 0.6
        self.epsilon = 1e-2
        self.__device = device
        self.__size = 0
        self.__pos = 0
        self.abs_error_upper = 1
        self.__capacity = capacity
        self.position = 0
        self.n_multi_step = 3
        self.gamma = gamma

        """height of the sumtree"""
        self.layer = math.ceil(math.log(capacity, 2))
        """nodes in the sumtree"""
        self.buffer_capacity = 2**(self.layer + 1) + 1
        """probability in the sumtree"""
        self.buffer = np.zeros(self.buffer_capacity)
        """start position of leaf"""
        self.leaf_start = self.buffer_capacity // 2

        self.memory = np.array([None]*self.buffer_capacity)

    def push(self, *args):
        """Saves a transition."""
        """the new state holds the max probability"""
        p = np.max(self.buffer[self.leaf_start:])
        """set the probability to the abs_error_upper"""
        if p == 0:
            p = self.abs_error_upper

        """"insert position in the sumtree"""
        buffer_pos = self.leaf_start + self.__pos

        """insert record"""
        self.memory[buffer_pos] = Transition(*args)

        """update the probability"""
        self.update(buffer_pos, p)

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def update(self, x, p):
        """update the probability"""
        """calculate the change"""
        change = p - self.buffer[x]
        self.buffer[x] = p

        """update the father node"""
        while x != 1:
            x = x // 2
            self.buffer[x] += change

    """find the leaf correspending to probability p"""

    def retrieve(self, x, p):
        if x >= self.leaf_start and x <= self.buffer_capacity - 1:
            return x
        lson = int(x * 2)
        rson = int(x * 2 + 1)
        if p <= self.buffer[lson]:
            return self.retrieve(lson, p)
        else:
            return self.retrieve(rson, p - self.buffer[lson])

    """the total probability"""

    def total_p(self):
        return self.buffer[1]

    def sample(self, batch_size):
        """sample record"""
        """the length of each interval is pri_seg to ensure sample uniformly"""
        pri_seg = self.total_p() / batch_size

        """random probability array"""
        sample_p = np.array(
            [np.random.uniform(pri_seg * i, pri_seg * (i + 1)) for i in range(batch_size)])

        """find the leaf nodes"""
        indices = np.array([self.retrieve(1, p) for p in sample_p])

        """get the data"""
        return self.memory[indices]

    def nstep_sample(self, batch_size):
        """n-step reward"""
        pri_seg = self.total_p() / batch_size

        """random probability array"""
        sample_p = np.array(
            [np.random.uniform(pri_seg * i, pri_seg * (i + 1)) for i in range(batch_size)])

        """find the leaf nodes"""
        indices = np.array([self.retrieve(1, p) for p in sample_p])
        batch = []
        for i in range(batch_size):
            begin = indices[i]
            finish = min(begin+self.n_multi_step, self.leaf_start+self.__size)
            sum_reward = 0  # n_step rewards
            data = self.memory[begin:finish]
            state = data[0].state
            action = data[0].action
            for j in range(0, finish-begin):
                # compute the n-th reward
                sum_reward += (self.gamma**j) * data[j].reward
                if not data[j].mask:
                    # manage end of episode
                    next_state = data[j].state
                    mask = data[j].mask
                    break
                else:
                    next_state = data[j].state
                    mask = data[j].mask

            batch.append(Transition(
                state, action, mask, next_state, sum_reward))
        return batch

    def __len__(self) -> int:
        return self.__size

    """"update the probability after training"""

    def batch_update(self, b_idx, abs_errors):
        """ensure abs_errors>0"""
        abs_errors += self.epsilon

        """ensure abs_errors<=abs_error_upper"""
        abs_errors[abs_errors > self.abs_error_upper] = self.abs_error_upper

        """power"""
        ps = np.power(abs_errors, self.alpha)

        """update"""
        for idx, p in zip(b_idx, ps):
            self.update(idx, p)
