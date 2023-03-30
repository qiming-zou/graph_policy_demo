import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, G, hidden_dim=128):
        super().__init__()

        self.s_enc = G["encoder"]["s_enc"]
        self.g_enc = G["encoder"]["g_enc"]
        a_dim = len(G["act_set"])
        self.goal_id_to_goal = G["goal_id_to_goal"]
        self.state_id_to_goal = G["state_id_to_goal"]

        self.net = nn.Sequential(
            nn.Linear(self.s_enc.z_dim + self.g_enc.z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, a_dim)
        )
