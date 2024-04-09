import math
import pdb

import torch


def calc(level, sigema, x, y=0):
    if level == 1:
        return (
            1
            / ((2 * math.pi) ** 0.5 * sigema)
            * math.exp(-(x**2 / 2 / (sigema**2)))
        )
    elif level == 2:
        return (
            1
            / (2 * math.pi * sigema * sigema)
            * math.exp(-(x**2 + y**2) / 2 / sigema / sigema)
        )


def apply_gussian(pose):
    """
    pose: [length, 6],torch tensor
    """
    length = pose.shape[0]
    window_size = 29
    zeros = torch.zeros(window_size // 2, 3).to(pose.device)
    pose = torch.cat((zeros, pose, zeros), dim=0)
    result = torch.zeros(length, 3).to(pose.device)
    for i in range(length):
        p1 = 0
        p2 = 0
        p3 = 0
        for j in range(-window_size // 2, window_size // 2 + 1):
            p1 += pose[14 + i + j, 0] * calc(level=1, sigema=2, x=j)
            p2 += pose[14 + i + j, 1] * calc(level=1, sigema=2, x=j)
            p3 += pose[14 + i + j, 2] * calc(level=1, sigema=2, x=j)
        result[i, 0] = p1
        result[i, 1] = p2
        result[i, 2] = p3
    return result
