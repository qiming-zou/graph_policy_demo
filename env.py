import gym
import networkx as nx
from copy import deepcopy
import random
import pickle
import numpy as np
from save_graph import pickle_load




class Env(gym.Env):

    def __init__(self, G, mode):
        super().__init__()
        self.T = 100
        self.mode = mode
        self.graph = deepcopy(G["graph"])
        self.goal_id_to_goal = deepcopy(G["goal_id_to_goal"])
        self.action_space = gym.spaces.Discrete(n=G["n_act"])
        self.observation_space = gym.spaces.Box(
            low=np.asarray([-np.inf] * 3), high=np.asarray([np.inf] * 3), shape=(3,)
        )

    def reset(self, goal_id=None):
        self.t = 1
        self.node_t = random.choice(list(self.graph.nodes)) if self.mode=="train" else list(self.graph.nodes)[0]
        if goal_id is None:
            self.graph_seed = random.choice(list(self.goal_id_to_goal.graph_seed_to_goal_id.keys()))
            self.goal_id = random.choice(list(self.goal_id_to_goal.graph_seed_to_goal_id[self.graph_seed]))
        else:
            self.goal_id = goal_id
            self.graph_seed = random.choice([graph_seed for graph_seed in self.goal_id_to_goal.graph_seed_to_goal_id.keys() if goal_id in self.goal_id_to_goal.graph_seed_to_goal_id[graph_seed] ])
            can_success = False
            for node in  self.graph.nodes:
                if self.graph_seed in self.graph.nodes[node]["achieved_goal"]:
                    if self.goal_id in self.graph.nodes[node]["achieved_goal"][self.graph_seed]:
                        can_success = True
            assert can_success

        return {"state_id": self.node_t, "goal_id": self.goal_id}

    def step(self, act_id):
        self.t += 1
        self.node_t = self.graph.nodes[self.node_t][act_id]

        timeout = self.t >= self.T
        success = (self.graph_seed in self.graph.nodes[self.node_t]["achieved_goal"].keys()) and (self.goal_id in self.graph.nodes[self.node_t]["achieved_goal"][self.graph_seed])
        reward = float(success)
        done = (success) or (timeout)

        return {"state_id": self.node_t, "goal_id": self.goal_id}, reward, done, {}


def make_fn(env_name, mode):
    G = pickle_load(path=f"{env_name}/graph.pkl")
    return Env(G=G, mode=mode)


if __name__ == "__main__":
    env = make_fn(env_name="ai2thor_obj_search")
    for i in range(10000):
        env.reset()
        done = False
        while not done:
            act = env.action_space.sample()
            state, reward, done, _ = env.step(act_id=act)
        if env.t < env.T:
            print(env.t, state,env.graph.nodes[state["state_id"]]["achieved_goal"][env.graph_seed], reward)
