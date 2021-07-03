import numpy as np
import torch


def seed_random(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
