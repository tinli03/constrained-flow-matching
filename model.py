# defines the network
import torch
from torch import nn


class FlowNetwork(nn.Module):
    # 3 hidden layer network, hidden width = 128, activation SiLU
    def __init__(self, args, n_dim = 10):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(n_dim + 1, args.hidden_width), nn.SiLU(),
        nn.Linear(args.hidden_width, args.hidden_width), nn.SiLU(),
        nn.Linear(args.hidden_width, args.hidden_width), nn.SiLU(),
        nn.Linear(args.hidden_width, n_dim),
        )

    def forward(self, xt, t):
        inp = torch.cat([xt, t.unsqueeze(1)], dim=1)
        return self.net(inp)


