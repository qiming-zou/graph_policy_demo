import torch
import torch.nn as nn
import torch.nn.functional as F
from zqmtool.ssl import reparameterization

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim*2)
        )

    def forward(self, x):
        x = self.net(x)
        z, mu, sigma = reparameterization(params=x, z_dim=self.z_dim)
        return z, mu, sigma

