# runs the training and calls model and data
import torch
import argparse
import numpy as np
from input import sample_training_batch
from model import FlowNetwork
from source_to_file import N_DIM 
from utils import to_tensor
import matplotlib.pyplot as plt

def parse_args():
    # parses the command line args for the model
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_width", type=int, default=128)
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--rmseed", type=int, default=0)
    return parser.parse_args()

def plot_loss(loss_iters, loss_values, val_iters, val_losses):
    plt.figure()
    plt.plot(loss_iters, loss_values, label="training loss")
    plt.plot(val_iters, val_losses, label="validation loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()

def train(args, model, optimizer):
    # TODO ROSE draw some validation data 
    loss_iters = []
    loss_values = []
    val_iters = []
    val_losses = []

    for i in range(1, args.n_iters + 1):
        xt, t, target = sample_training_batch(args.batch_size)  

        xt = to_tensor(xt)
        t = to_tensor(t)
        target = to_tensor(target)

        pred = model(xt, t)                                       
        loss = ((pred - target) ** 2).mean()                      

        optimizer.zero_grad()                                      
        loss.backward()                                              
        optimizer.step()     

        if i % args.val_every == 0:
            print(f"iter {i}: loss = {loss.item():.5f}")
            loss_iters.append(i)
            loss_values.append(loss.item())
    
        if i % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_xt, val_t)
                val_loss = ((val_pred - val_target) ** 2).mean()
            model.train()
            print(f"iter {i}: val_loss = {val_loss.item():.5f}")
            val_iters.append(i)          
            val_losses.append(val_loss.item())   

    plot_loss(loss_iters, loss_values, val_iters, val_losses)
    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    args = parse_args()

    np.random.seed(12345)                             
    val_xt, val_t, val_target = sample_training_batch(args.val_size)
    val_xt = to_tensor(val_xt)
    val_t = to_tensor(val_t)
    val_target = to_tensor(val_target)

    set_seed(args.rmseed)
    model = FlowNetwork(args, N_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(args, model, optimizer)


