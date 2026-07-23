# inferring using trained model
import torch
import numpy as np
from input import sample_training_batch
from model import FlowNetwork
from source_to_file import N_DIM 
from utils import to_tensor
import matplotlib.pyplot as plt

model = FlowNetwork(args, N_DIM)
model.load_state_dict(torch.load(args.checkpoint_path))
model.eval()

def sample_unconstrained(model, x0, n_steps=100):
    x = x0
    dt = 1.0 / n_steps # hur lång tid varje steg är
    for k in range(n_steps):
        t = torch.full((x.shape[0],), k * dt) # hur långt tidsmässigt vi har kommit fram
        v = model(x, t)
        x = x + dt * v
    return x