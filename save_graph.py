import os.path

import networkx as nx
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from collections import defaultdict
from encoder import MLP
from zqmtool.other import to_np
from scipy.spatial.distance import cdist


def pickle_load(path):
    env_name = os.path.dirname(path)
    assert env_name in ["maze"]
    if env_name == "maze":
        from maze.def_graph import StateID2State, GoalID2Goal, StateID2AchievedGoalID, StateIDtoNextStateID
    else:
        raise NotImplementedError
    return pickle.load(open(path, "rb"))


env_name = "maze"
if env_name == "maze":
    from maze.def_graph import StateID2State, GoalID2Goal, StateID2AchievedGoalID, StateIDtoNextStateID
else:
    raise NotImplementedError

G = nx.Graph()

# state
state_id_to_state = StateID2State()
G.add_nodes_from(state_id_to_state.id_set)
for node in G.nodes:
    assert isinstance(node, int)

# state transition
state_id_to_next_state_id = StateIDtoNextStateID(state_id_to_state=state_id_to_state)
for node in G.nodes:
    for act_id in range(len(state_id_to_next_state_id.act_set)):
        next_state_id = state_id_to_next_state_id(state_id=node, act_id=act_id)
        G.nodes[node][act_id] = next_state_id

# goal
goal_id_to_goal = GoalID2Goal()
state_id_to_achieved_goal_id = StateID2AchievedGoalID(
    state_id_to_state=state_id_to_state,
    goal_id_to_goal=goal_id_to_goal
)
for node in G.nodes:
    G.nodes[node]["achieved_goal"] = state_id_to_achieved_goal_id(node)
    assert isinstance(G.nodes[node]["achieved_goal"], set)

# encoder
s_enc = MLP(in_dim=2)
g_enc = MLP(in_dim=10)

pickle.dump(
    {
        "graph": G,
        "goal_id_to_goal": goal_id_to_goal,
        "state_id_to_state": state_id_to_state,
        "state_id_to_achieved_goal_id": state_id_to_achieved_goal_id,
        "n_act": len(state_id_to_next_state_id.act_set),
        "encoder": {"s_enc": s_enc, "g_enc": g_enc}
    }, open(f"{env_name}/graph.pkl", "wb")
)
