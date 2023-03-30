import gym
import torch, numpy as np, torch.nn as nn
import torch.cuda
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from env import make_fn
from graph import pickle_load

from tianshou.utils.net.common import Net
import pickle
from policy import Policy

lr, epoch, batch_size = 1e-3, 10, 64
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, train_num
logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# you can also try with SubprocVectorEnv
train_envs = ts.env.DummyVectorEnv([lambda: make_fn() for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: make_fn() for _ in range(test_num)])

G = pickle_load(path="graph.pkl")
net = Policy(G=G, device=device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
    test_num, batch_size, update_per_step=1 / step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    logger=logger)
print(f'Finished training! Use {result["duration"]}')