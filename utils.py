import torch
import numpy as np

def to_tensor(x):
    return torch.from_numpy(np.asarray(x)).float()