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