#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :main.py
# @Time      :2021/8/6 下午3:34
# @Author    :miracleyin
# @Email     :miracleyin@live.com
import time
import torch
from utils import setup_seed
from env import DGEnv
from agent import QLearning


class QlearningConfig:
    def __init__(self):
        self.algo = 'Qlearning'
        self.env = DGEnv()
        self.result_path = f'./output/{self.algo}/result/'
        self.model_path = f'./output/{self.algo}/model/'
        self.train_eps = 100
        self.eval_eps = 30
        self.gamma = 0.9
        self.epsilon_start = 0.95  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 200  # e-greedy策略中epsilon的衰减率
        self.lr = 0.1  # learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check gpu

class DQNConfig:
    def __init__(self):
        self.algo = "DQN"  # name of algo
        self.env = 'DGEnv'
        self.result_path = f'./output/{self.algo}/result/'
        self.model_path = f'./output/{self.algo}/model/'
        self.train_eps = 300  # max trainng episodes
        self.eval_eps = 50  # number of episodes for evaluating
        self.gamma = 0.95
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 100000  # capacity of Replay Memory
        self.batch_size = 64
        self.target_update = 4  # update frequency of target net
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu
        self.hidden_dim = 256  # hidden size of net

def env_agent_config(cfg, seed=42):
    env = DGEnv()
    # env.get_config({}) # 用于更新环境参数
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = QLearning(state_dim, action_dim, cfg)
    return env, agent


def main():
    cfg = QlearningConfig()
    env, agent = env_agent_config(cfg)


if __name__ == '__main__':
    main()
