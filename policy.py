import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import MLP, Net
from zqmtool.other import to_np, to_th


class Policy(nn.Module):
    def __init__(self, G, device, hidden_dim=128):
        super().__init__()
        self.device = device
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

        self.to(device=device)

    def forward(self, obs, state, info):
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        s = to_th(self.state_id_to_goal(obs["state_id"]), device=self.device)
        g = to_th(self.goal_id_to_goal(obs["goal_id"]), device=self.device)
        z_s, *_ = self.s_enc(s)
        z_g, *_ = self.g_enc(g)
        z = torch.cat([z_s, z_g], dim=-1)
        logits = self.net(z)
        return logits, state
