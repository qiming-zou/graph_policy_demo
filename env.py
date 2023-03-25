import gym
import networkx as nx
from copy import deepcopy
import random
import pickle
import numpy as np

G = pickle.load(open("graph.pkl", "rb"))

class Env(gym.Env):

    def __init__(self):
        super().__init__()
        self.T = 10
        self.graph = deepcopy(G["graph"])
        self.act_set = deepcopy(G["act_set"])
        w, h = G["maze_size"]
        self.action_space = gym.spaces.Discrete(n=len(self.act_set))
        self.observation_space = gym.spaces.Box(
            low=np.asarray([0, 0, 0, 0]), high=np.asarray([w, h, w, h]), shape=(4,)
        )

    def reset(self, **kwargs):
        self.t = 1
        self.node_t = list(self.graph.nodes)[0]
        self.node_goal = random.choice(list(self.graph.nodes))
        state = list(self.node_t) + list(self.node_goal)
        return state

    def step(self, act_id):
        self.t += 1
        action = self.act_set[act_id]
        self.node_t = self.graph.nodes[self.node_t][action]
        state = list(self.node_t) + list(self.node_goal)

        timeout = self.t >= self.T
        success = self.node_t == self.node_goal
        reward = float(success)
        done = (success) or (timeout)

        return state, reward, done, {}

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
            print(env.t, state, reward)
