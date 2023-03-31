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


class StateID2State(ID2Info):
    def __init__(self):
        super().__init__()
        w = h = 10
        state_set = [[xi, xj] for xi in range(h) for xj in range(w)]
        self.id_set = list(range(len(state_set)))
        self.id_to_info_np = to_np(state_set)
        self.test()


class GoalID2Goal(ID2Info):
    def __init__(self):
        super().__init__()
        self.id_set = list(range(10))
        self.id_to_info_np = to_np([to_onehot(goal_id, 10) for goal_id in self.id_set])


class StateID2AchievedGoalID():
    def __init__(
            self,
            state_id_to_state: StateID2State,
            goal_id_to_goal: GoalID2Goal,
    ):
        self.state_id_to_state = state_id_to_state
        self.goal_id_to_goal = goal_id_to_goal

    def __call__(self, state_id: int):
        assert isinstance(state_id, int)
        state = self.state_id_to_state(state_id)
        achieved_goal_id = {random.choice(self.goal_id_to_goal.id_set)} if random.random()<0.5 else set()
        return achieved_goal_id


class StateIDtoNextStateID():
    def __init__(self, state_id_to_state: StateID2State):
        self.state_id_to_state = state_id_to_state
        self.act_set = ["left", "right", "up", "down"]

    def __call__(self, state_id: int, act_id: int):
        assert isinstance(state_id, int)
        assert isinstance(act_id, int)
        assert (act_id >= 0) and (act_id < len(self.act_set))
        act = self.act_set[act_id]
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

        state = self.state_id_to_state(state_id)
        state_next = state.copy()
        state_next[0] = state_next[0] + step[0]
        state_next[1] = state_next[1] + step[1]

        min_d = 1e8
        best_id = None
        for elem in self.state_id_to_state.id_set:
            elem_state = self.state_id_to_state(elem)
            d = np.linalg.norm(elem_state - state_next)
            if d < min_d:
                min_d = d
                best_id = elem

        if min_d < 1e-4:
            return best_id
        else:
            return state_id


def to_onehot(v, n):
    out = np.zeros(n)
    out[v] = 1
    return out
