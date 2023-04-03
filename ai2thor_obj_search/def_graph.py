import glob
import os.path

import networkx as nx
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from collections import defaultdict
from zqmtool.other import to_np
from scipy.spatial.distance import cdist
from tqdm import trange
from copy import deepcopy

cwd = os.path.dirname(__file__)


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
        info_np = deepcopy(self.id_to_info_np[id_np])
        return info_np


class StateID2State(ID2Info):
    def __init__(self):
        super().__init__()
        self.nav_graph = pickle.load(open(os.path.join(cwd, "G/FloorPlan1.pickle"), "rb"))

        self.state_lst = []
        self.id_set = []
        for state_id, state in enumerate(self.nav_graph.nodes):
            self.state_lst.append(node2state(state))
            self.id_set.append(state_id)

        # normalization
        state_np = to_np(self.state_lst)
        s_min = np.amin(state_np, axis=0)
        s_ptp = np.ptp(state_np, axis=0)
        self.id_to_info_np = (state_np - s_min) / s_ptp

    def reverse_mapping(self, state):
        state_id = self.state_lst.index(state)
        return state_id


class GoalID2Goal(ID2Info):
    def __init__(self):
        super().__init__()

        # all_goal = set()
        # for graph_seed in trange(10):
        #     graph_file = os.path.join(cwd, f"G/FloorPlan1_{graph_seed}.pickle")
        #     G = pickle.load(open(graph_file, "rb"))
        #     names = set()
        #     for node in G.nodes:
        #         names.update(set([obj["name"] for obj in G.nodes[node]["object"].values()]))
        #     all_goal.update(set([name+f"_seed{graph_seed}" for name in names]))
        # print(all_goal)


        # goal_name_2_graph_file = defaultdict(set)
        # for graph_seed in trange(10):
        #     graph_file = os.path.join(cwd, f"G/FloorPlan1_{graph_seed}.pickle")
        #     G = pickle.load(open(graph_file, "rb"))
        #     for node in G.nodes:
        #         objs_in_node = G.nodes[node]["object"].values()
        #         for obj in objs_in_node:
        #             goal_name_2_graph_file[obj["name"]].add(graph_seed)
        #
        # self.id_to_info_np = []
        # self.id_set = []
        # id = 0
        # for goal_name in goal_name_2_graph_file.keys():
        #     for graph_seed in goal_name_2_graph_file[goal_name]:
        #         self.id_set.append(id)
        #         self.id_to_info_np.append(goal_name)
        #         id += 1

        # self.id_to_vec_info_lst = []
        # info_set = sorted(set(self.id_to_info_np))
        # for id in self.id_set:
        #     info = self.id_to_info_np[id]
        #     info_index = info_set.index(info)
        #     vec = np.zeros(len(info_set))
        #     vec[info_index] = 1
        #     self.id_to_vec_info_lst.append(vec)

    def reverse_mapping(self, goal):
        goal_id = self.id_to_info_np.index(goal)
        return goal_id


class StateID2AchievedGoalID():
    def __init__(
            self,
            state_id_to_state: StateID2State,
            goal_id_to_goal: GoalID2Goal,
    ):
        self.state_id_to_state = state_id_to_state
        self.goal_id_to_goal = goal_id_to_goal

        self.state_id_2_achieved_goal_id = defaultdict(dict)
        for graph_seed in trange(10):
            graph_file = os.path.join(cwd, f"G/FloorPlan1_{graph_seed}.pickle")
            G = pickle.load(open(graph_file, "rb"))
            for node in G.nodes:
                state_id = self.state_id_to_state.reverse_mapping(node2state(node))
                objs_in_node = G.nodes[node]["object"].values()
                for obj in objs_in_node:
                    if obj["name"] in self.goal_id_to_goal.id_to_info_np:
                        goal_id = self.goal_id_to_goal.reverse_mapping(obj["name"])
                        if graph_seed not in self.state_id_2_achieved_goal_id[state_id]:
                            self.state_id_2_achieved_goal_id[state_id][graph_seed] = set()
                        self.state_id_2_achieved_goal_id[state_id][graph_seed].add(goal_id)

    def __call__(self, state_id: int):
        return self.state_id_2_achieved_goal_id[state_id]


class StateIDtoNextStateID():
    def __init__(self, state_id_to_state: StateID2State):
        self.state_id_to_state = state_id_to_state
        self.act_set = set()
        self.nav_graph = pickle.load(open(os.path.join(cwd, "G/FloorPlan1.pickle"), "rb"))

        self.state_id_and_act_to_next_state_id = defaultdict(dict)
        self.state_id_availabel_act = defaultdict(set)
        for node1, node2, data in self.nav_graph.edges(data=True):
            act = data["act"]
            if act.find("Open") != -1:
                act = "Open"
            elif act.find("Close") != -1:
                act = "Close"
            self.act_set.add(act)

            state_id1 = self.state_id_to_state.reverse_mapping(node2state(node1))
            state_id2 = self.state_id_to_state.reverse_mapping(node2state(node2))
            self.state_id_and_act_to_next_state_id[state_id1][act] = state_id2
            self.state_id_availabel_act[state_id1].add(act)

        self.act_set = sorted(list(self.act_set))

    def __call__(self, state_id: int, act_id: int):
        act = self.act_set[act_id]
        if act not in self.state_id_availabel_act[state_id]:
            return state_id
        next_state_id = self.state_id_and_act_to_next_state_id[state_id][act]
        return next_state_id


def node2state(node):
    node = eval(node)
    state = [node["position"]["x"], node["position"]["z"], node["rotation"]["y"]]
    return state


if __name__ == "__main__":
    stateid2state = StateID2State()
    # goalid2goal = GoalID2Goal()
    # stateid2achievedgoalid = StateID2AchievedGoalID(
    #     state_id_to_state=stateid2state, goal_id_to_goal=goalid2goal
    # )
    stateid2nextstateid = StateIDtoNextStateID(state_id_to_state=stateid2state)
    stateid2nextstateid(0, 0)
