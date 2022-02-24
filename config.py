#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :config.py
# @Time      :2021/9/6 下午9:22
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import os
import sys
import datetime
import torch

class ENVConfig:
    def __init__(self):
        self.dog_spd = 10
        self.dog_siz = 30
        self.goat_spd = 10
        self.goat_siz = 15

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time

class SIMPLEConfig:
    def __init__(self):
        self.algo = "SIMPLE"
        self.env = "Env_simple"
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + curr_time + '/results/'
        self.model_path = curr_path + "outputs/" + self.env + \
                          '/' + curr_time + '/models/'


class DQNConfig:
    def __init__(self):
        self.algo = "DQN"
        self.env = 'DGEnv'  # for test algorithmics
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + curr_time + '/results/'
        self.model_path = curr_path + "outputs/" + self.env + \
                          '/' + curr_time + '/models/'
        # model set
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