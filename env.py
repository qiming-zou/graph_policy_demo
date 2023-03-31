import gym
import networkx as nx
from copy import deepcopy
import random
import pickle
import numpy as np
from def_graph import GoalID2Goal

G = pickle.load(open("graph.pkl", "rb"))


class Env(gym.Env):

    def __init__(self):
        super().__init__()
        self.T = 100
        self.graph = deepcopy(G["graph"])
        self.goal_id_to_goal = deepcopy(G["goal_id_to_goal"])
        self.action_space = gym.spaces.Discrete(n=G["n_act"])
        self.observation_space = gym.spaces.Box(
            low=np.asarray([-np.inf] * 3), high=np.asarray([np.inf] * 3), shape=(3,)
        )

    def reset(self, **kwargs):
        self.t = 1
        self.node_t = list(self.graph.nodes)[0]
        self.goal_id = random.choice(self.goal_id_to_goal.id_set)
        return {"state_id": self.node_t, "goal_id": self.goal_id}

    def step(self, act_id):
        self.t += 1
        self.node_t = self.graph.nodes[self.node_t][act_id]

        timeout = self.t >= self.T
        success = self.goal_id in self.graph.nodes[self.node_t]["achieved_goal"]
        reward = float(success)
        done = (success) or (timeout)

        return {"state_id": self.node_t, "goal_id": self.goal_id}, reward, done, {}


def make_fn():
    return Env()


if __name__ == "__main__":
    env = Env()
    for i in range(10000):
        env.reset()
        done = False
        while not done:
            act = env.action_space.sample()
            state, reward, done, _ = env.step(act_id=act)
        if env.t < env.T:
            print(env.t, state,env.graph.nodes[state["state_id"]]["achieved_goal"], reward)
