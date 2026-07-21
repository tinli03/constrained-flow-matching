# runs the training and calls model and data
import torch
import argparse

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
    for i in range(1, args.n_iters + 1):
        xt, t, target = sample_training_batch(args.batch_size, ...)  # TODO
        pred = model(xt, t)                                       
        loss = ((pred - target) ** 2).mean()                      

        optimizer.zero_grad()                                      
        loss.backward()                                              
        optimizer.step()  


if __name__ == "__main__":
    args = parse_args()
    model = FlowNetwork(args, n_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(args, model, optimizer)


