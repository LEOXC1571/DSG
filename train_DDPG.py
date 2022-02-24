#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :train_DDPG.py
# @Time      :2021/9/5 下午3:44
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com


import sys, os
import time

from env import DGEnv

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from agent import DDPG
from env import NormalizedActions, OUNoise
import gym
from utils import save_results, make_dir, plot_rewards, plot_losses

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time


class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'
        self.env = 'DGEnv'  # env name
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + self.algo + '/' + curr_time + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env + \
                          '/' + self.algo + '/' + curr_time + '/models/'
        self.gamma = 0.99
        self.critic_lr = 1e-3
        self.actor_lr = 1e-4
        self.memory_capacity = 10000
        self.batch_size = 128
        self.train_eps = 300
        self.eval_eps = 50
        self.eval_steps = 200
        self.target_update = 4
        self.hidden_dim = 30
        self.soft_tau = 1e-2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ENVConfig:
    def __init__(self):
        self.dog_spd = 20
        self.dog_siz = 30
        self.goat_spd = 10
        self.goat_siz = 15


def env_agent_config(env_cfg, agent_cfg, seed=42):
    env = NormalizedActions(DGEnv(env_cfg, is_continue=True))
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim, action_dim, agent_cfg)
    return env, agent


def train(cfg, env, agent, is_display=False):
    print('Start to train ! ')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    ou_noise = OUNoise(env.action_space)
    rewards = []
    ma_rewards = []  # moving average rewards
    train_times = 0
    for i_ep in range(cfg.train_eps):
        t1 = time.time()
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, _ = env.step(action)
            if is_display:
                env.render()
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state

        if (i_ep + 1) % 10 == 0:
            print(f'Episode:{i_ep + 1}/{agent_cfg.train_eps}, Reward:{ep_reward}')

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        train_times += time.time() - t1
    print('Complete training！')
    return rewards, ma_rewards, train_times/300


def eval(cfg, env, agent, is_display=False):
    print('Start to Eval ! ')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    std_actions = 0
    actionss = []
    for i_ep in range(cfg.eval_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            actionss.append(action)
            next_state, reward, done, _ = env.step(action)
            if is_display:
                env.render()
            ep_reward += reward
            state = next_state
        print(f'Episode:{i_ep + 1}/{cfg.eval_eps}, Reward:{ep_reward}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        temp_action = np.array(actionss)
        std_actions += np.std(temp_action)
        actionss = []

    print('Complete Eval！')
    return rewards, ma_rewards,std_actions


def random_test(cfg, env, is_display=False):
    print('Start to BASELINE RANDOM! ')
    print(f'Env:{cfg.env}, Algorithm: RANDOM, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    gr = []
    for i_ep in range(cfg.eval_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = env.action_space.sample()
            next_state, reward, done, winner = env.step(action)
            if is_display:
                env.render()
            ep_reward += reward
            state = next_state
        print(f'Episode:{i_ep + 1}/{cfg.eval_eps}, Reward:{ep_reward}')
        rewards.append(ep_reward)
        gr.append(winner)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete Eval！')
    return rewards, ma_rewards, gr


if __name__ == "__main__":
    agent_cfg = DDPGConfig()
    env_cfg = ENVConfig()
    # # train
    # env, agent = env_agent_config(env_cfg, agent_cfg, seed=1)
    # rewards, ma_rewards = train(agent_cfg, env, agent)
    make_dir(agent_cfg.result_path, agent_cfg.model_path)
    # agent.save(path=agent_cfg.model_path)
    # save_results(rewards, ma_rewards, tag='train', path=agent_cfg.result_path)
    # plot_rewards(rewards, ma_rewards, tag="train",
    #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)
    #
    # # eval
    # env, agent = env_agent_config(env_cfg, agent_cfg, seed=10)
    # agent.load(path=agent_cfg.model_path)
    # rewards, ma_rewards = eval(agent_cfg, env, agent)
    # save_results(rewards, ma_rewards, tag='eval', path=agent_cfg.result_path)
    # plot_rewards(rewards, ma_rewards, tag="eval",
    #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)

    # baseline random
    # env = DGEnv(env_cfg, is_continue=True)
    # rewards, ma_rewards = random_test(agent_cfg, env, is_display=True)
    # save_results(rewards, ma_rewards, tag='baseline', path=agent_cfg.result_path)
    # plot_rewards(rewards, ma_rewards, tag="baseline",
    #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)
    from sklearn.metrics import mean_squared_error

    mse = []
    target = 5 * np.ones(shape=(40, 1))
    ttt = []
    std_a = []

    for idx, spd in enumerate(range(10, 51, 10)):
        env_cfg.dog_spd = spd
        env, agent = env_agent_config(env_cfg, agent_cfg, seed=42)
        _, _, tt = train(agent_cfg, env, agent, is_display=False)
        make_dir(agent_cfg.result_path, agent_cfg.model_path)
        agent.save(path=agent_cfg.model_path)
        ttt.append(tt)

        env, agent = env_agent_config(env_cfg, agent_cfg, seed=1000)
        agent.load(path=agent_cfg.model_path)
        rewards, ma_rewards, std_actions = eval(agent_cfg, env, agent)
        save_results(rewards, ma_rewards, tag='eval', path=agent_cfg.result_path)
        pred = np.array(rewards[10:])
        mse.append(mean_squared_error(pred, target))
        std_a.append(std_actions)
    print(mse)
    print(ttt)
    print(std_a)



    #     # _, _ = train(agent_cfg, env, agent, is_display=False)
    #     # make_dir(agent_cfg.result_path, agent_cfg.model_path)
    #     # agent.save(path=agent_cfg.model_path)
    #     #
    #     env, agent = env_agent_config(env_cfg, agent_cfg, seed=10)
    #     # agent.load(path=agent_cfg.model_path)
    #     rewards, ma_rewards, game_results = random_test(agent_cfg, env)
    #     delta.append(game_results.count(0) / 50)
    # print(delta)
