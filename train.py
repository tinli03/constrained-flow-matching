# runs the training and calls model and data
import torch
import argparse
import numpy as np
from input import sample_training_batch
from model import FlowNetwork
from data import n_dims # rose skriv in från datan

def parse_args():
    # parses the command line args for the model
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_width", type=int, default=128)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--rmseed", type=int, default=0)
    return parser.parse_args()

def train(args, model, optimizer):
    # TODO ROSE draw some validation data 
    val_rng = np.random.default_rng(12345)
    for i in range(1, args.n_iters + 1):
        xt, t, target = sample_training_batch(i)  # TODO Rose
        pred = model(xt, t)                                       
        loss = ((pred - target) ** 2).mean()                      

        optimizer.zero_grad()                                      
        loss.backward()                                              
        optimizer.step()  
        
    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.rmseed)
    model = FlowNetwork(args, n_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(args, model, optimizer)


