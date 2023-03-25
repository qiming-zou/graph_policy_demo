import networkx as nx
import matplotlib.pyplot as plt
import pickle
import random

w = h = 10
act_set = ["left", "right", "up", "down"]
goal_set = list(range(10))
G = nx.Graph()

nodes = [(xi, xj) for xi in range(h) for xj in range(w)]
G.add_nodes_from(nodes)

for node in nodes:
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
        node_next = list(node)
        node_next[0] = min(max(node_next[0] + step[0], 0), h - 1)
        node_next[1] = min(max(node_next[1] + step[1], 0), w - 1)
        node_next = tuple(node_next)
        G.add_edge(node, node_next, name=act)
        G.nodes[node][act] = node_next
        G.nodes[node]["achieved_goal"] = random.choice(goal_set) if random.random() < 0.2 else None

        # G[node]["act"] = node_next

pickle.dump({
    "graph": G, "act_set": act_set, "goal_set": goal_set, "maze_size": (w, h)},
    open("graph.pkl", "wb")
)
