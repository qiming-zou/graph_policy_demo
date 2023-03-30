import networkx as nx
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from collections import defaultdict
from encoder import MLP
from zqmtool.other import to_np
from scipy.spatial.distance import cdist


class ID2Info():
    def __init__(self, ):
        self.id_set = None
        self.id_to_info_np = None

    def test(self):
        assert self.id_set is not None
        assert self.id_to_info_np is not None
        assert len(self.id_set) == len(self.id_to_info_np)

    def __call__(self, id_np):
        id_np = to_np(id_np, dtype=int)
        info_np = self.id_to_info_np[id_np].copy()
        return info_np

    def reverse_mapping(self, info):
        info = to_np(info, dtype=np.float32)
        if len(info.shape)+1 == len(self.id_to_info_np.shape):
            info = info[None, ...]
        D = cdist(info, self.id_to_info_np)
        D = D.reshape(D.shape[0], -1)
        d_argmin = np.argmin(D, axis=-1)
        return d_argmin





def to_onehot(v, n):
    out = np.zeros(n)
    out[v] = 1
    return out


w = h = 10
act_set = ["left", "right", "up", "down"]

G = nx.Graph()


# state
class StateID2State(ID2Info):
    def __init__(self):
        super().__init__()
        state_set = [[xi, xj] for xi in range(h) for xj in range(w)]
        self.id_set = list(range(len(state_set)))
        self.id_to_info_np = to_np(state_set)
        self.test()

state_id_to_state = StateID2State()
G.add_nodes_from(state_id_to_state.id_set)


# state transition
for node in G.nodes:
    for act in act_set:
        if act == "left":
            step = [0, -1]
        elif act == "right":
            step = [0, 1]
        elif act == "up":
            step = [-1, 0]
        elif act == "down":
            step = [1, 0]
        else:
            raise NotImplementedError
        state_next = state_id_to_state(node)
        state_next[0] = min(max(state_next[0] + step[0], 0), h - 1)
        state_next[1] = min(max(state_next[1] + step[1], 0), w - 1)
        node_next = state_id_to_state.reverse_mapping(state_next)[0]
        G.add_edge(node, node_next, name=act)
        G.nodes[node][act] = node_next


# goal
class GoalID2Goal(ID2Info):
    def __init__(self):
        super().__init__()
        self.id_set = list(range(10))
        self.id_to_info_np = to_np([to_onehot(goal_id, 10) for goal_id in self.id_set])



goal_id_to_goal = GoalID2Goal()
for node in G.nodes:
    G.nodes[node]["achieved_goal"] = set()
    G.nodes[node]["achieved_goal"].add(random.choice(goal_id_to_goal.id_set) if random.random() < 0.2 else None)

# encoder
s_enc = MLP(in_dim=2)
g_enc = MLP(in_dim=10)

pickle.dump(
    {
        "graph": G,
        "act_set": act_set,
        "goal_id_to_goal": goal_id_to_goal,
        "state_id_to_goal": state_id_to_state,
        "maze_size": (w, h),
        "encoder": {"s_enc": s_enc, "g_enc": g_enc}
    }, open("graph.pkl", "wb")
)

def pickle_load(path):
    from graph import GoalID2Goal
    from graph import StateID2State
    return pickle.load(open(path, "rb"))

