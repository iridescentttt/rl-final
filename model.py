import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Actor(nn.Module):
    def __init__(self, infeat, outfeat, hfeat1, hfeat2, batchnorm):
        super(Actor, self).__init__()
        self.batchnorm = batchnorm

        self.fc1 = nn.Linear(infeat, hfeat1)
        if self.batchnorm:
            self.ln1 = nn.LayerNorm(hfeat1)

        self.fc2 = nn.Linear(hfeat1, hfeat2)
        if self.batchnorm:
            self.ln2 = nn.LayerNorm(hfeat2)

        self.fc3 = nn.Linear(hfeat2, outfeat)
        self.init_params()

    def init_params(self):
        """initialize parameters"""
        self.fc1.weight.data = myinit(
            self.fc1.weight.data.shape, fanin=True)
        self.fc2.weight.data = myinit(
            self.fc2.weight.data.shape, fanin=True)
        self.fc3.weight.data = myinit(
            self.fc3.weight.data.shape, fanin=False)

    def forward(self, x):
        if self.batchnorm:
            out = F.relu(self.ln1(self.fc1(x)))
            out = F.relu(self.ln2(self.fc2(out)))
        else:
            out = F.relu(self.fc1(x))
            out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out


class Critic(nn.Module):
    def __init__(self, infeat, outfeat, hfeat1, hfeat2, batchnorm):
        super(Critic, self).__init__()
        self.batchnorm = batchnorm

        self.fc1 = nn.Linear(infeat, hfeat1)
        if self.batchnorm:
            self.ln1 = nn.LayerNorm(hfeat1)
        self.fc2 = nn.Linear(hfeat1+outfeat, hfeat2)
        if self.batchnorm:
            self.ln2 = nn.LayerNorm(hfeat2)
            
        self.fc3 = nn.Linear(hfeat2, 1)
        self.init_params()

    def init_params(self):
        """initialize parameters"""
        self.fc1.weight.data = myinit(
            self.fc1.weight.data.shape, fanin=True)
        self.fc2.weight.data = myinit(
            self.fc2.weight.data.shape, fanin=True)
        self.fc3.weight.data = myinit(
            self.fc3.weight.data.shape, fanin=False)

    def forward(self, x, action):
        """return Q based on the state and action"""
        """analyse the state"""
        if self.batchnorm:
            out = F.relu(self.ln1(self.fc1(x)))
            """analyse the state and the action"""
            out = torch.cat((out, action), 1)
            out = F.relu(self.ln2(self.fc2(out)))
        else:
            out = F.relu(self.fc1(x))
            """analyse the state and the action"""
            out = torch.cat((out, action), 1)
            out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
