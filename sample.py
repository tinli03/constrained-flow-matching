# inferring using trained model
import torch
import numpy as np
from input import sample_training_batch
from model import FlowNetwork
from source_to_file import N_DIM 
from utils import to_tensor
import matplotlib.pyplot as plt
from train import parse_args
from input import sample_training_batch 

args = parse_args()
model = FlowNetwork(args, N_DIM)
model.load_state_dict(torch.load(args.checkpoint_path))



def sample_unconstrained(model, batch_size, n_steps):
    xt, _, _ = sample_training_batch(batch_size) 
    dt = 1.0 / n_steps # hur lång tid varje steg är
    x = to_tensor(xt)
    model.eval()

    with torch.no_grad():
        for k in range(n_steps):
            t = torch.full((x.shape[0],), k * dt, dtype=x.dtype,device=x.device) # hur långt tidsmässigt vi har kommit fram 
            v = model(x, t)
            x += dt * v
    return x

print(sample_unconstrained(model, 5, 5))