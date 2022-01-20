import torch
import numpy as np
import matplotlib.pyplot as plt


def myinit(size, fanin=True):
    """initialize parameters"""
    if fanin == True:
        v = 1. / np.sqrt(size[0])
    else:
        v = 3e-3
    return torch.Tensor(size).uniform_(-v, v)


def soft_update(target, source, tau):
    """soft update of the target network"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def visualize(q_step, reward_step, image_url):
    plt.figure(figsize=(5, 5))
    plt.axis("equal")
    x = np.array(reward_step)
    y = np.array(q_step)
    x_max = np.max(x)
    x_min = np.min(x)
    plt.xlim([x_min, x_max])
    plt.ylim([x_min, x_max])
    plt.plot([x_min, x_max], [x_min, x_max], color="grey")
    plt.scatter(x, y, color="grey", s=10)
    plt.grid(linestyle="-")
    plt.savefig(image_url, dpi=600, format='png')


