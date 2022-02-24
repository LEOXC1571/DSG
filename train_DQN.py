#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :train_DQN.py
# @Time      :2021/9/5 下午3:32
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import sys, os
import time

import numpy as np

from RL.env import DGEnv

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import gym
import torch
import datetime

from utils import save_results, make_dir
from utils import plot_rewards, plot_losses
from agent import DQN

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time


class DQNConfig:
    def __init__(self):
        self.algo = "DQN"
        self.env = 'DGEnv'  # for test algorithmics
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + self.algo + '/' + curr_time + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env + \
                          '/' + self.algo + '/' + curr_time + '/models/'
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


class ENVConfig:
    def __init__(self):
        self.dog_spd = 20
        self.dog_siz = 30
        self.goat_spd = 10
        self.goat_siz = 15


def env_agent_config(env_cfg, agent_cfg, seed=42):
    env = DGEnv(env_cfg, is_continue=False)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, action_dim, agent_cfg)
    return env, agent


def train(agent_cfg, env, agent, is_display=False):
    print('Start to train !')
    print(f'Env:{agent_cfg.env}, Algorithm:{agent_cfg.algo}, Device:{agent_cfg.device}')
    rewards = []
    ma_rewards = []  # moveing average reward
    loss_list = []
    train_time = 0

    for i_ep in range(agent_cfg.train_eps):
        t1 = time.time()
        state = env.reset()
        ep_reward = 0
        total_loss = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            if is_display:
                env.render()
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            loss = agent.update()  # 在没有到一个batch时候 没有loss
            if loss != 0:
                total_loss += loss.cpu().detach().numpy()
            if done:
                env.close()
                break
        train_time += time.time() - t1

        if (i_ep + 1) % agent_cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if (i_ep + 1) % 10 == 0:
            print(f'Episode:{i_ep + 1}/{agent_cfg.train_eps}, Reward:{ep_reward}')

        rewards.append(ep_reward)
        loss_list.append(total_loss)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

    print('Complete training！')
    return rewards, ma_rewards, loss_list, train_time / 300


def eval(agent_cfg, env, agent):
    print('Start to eval !')
    print(f'Env:{agent_cfg.env}, Algorithm:{agent_cfg.algo}, Device:{agent_cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    actions = []
    for i_ep in range(agent_cfg.eval_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.predict(state)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"Episode:{i_ep + 1}/{agent_cfg.eval_eps}, reward:{ep_reward:.1f}")

    print('Complete evaling！')
    return rewards, ma_rewards, actions


def simple_tset(cfg, env, is_display=False):
    print('Start to BASELINE SIMPLE! ')
    print(f'Env:{cfg.env}, Algorithm: SIMPLE, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    game_results = []
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
        game_results.append(winner)
        # print(f'Episode:{i_ep + 1}/{cfg.eval_eps}, Reward:{ep_reward}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete Eval！')
    return rewards, ma_rewards, game_results


if __name__ == "__main__":
    agent_cfg = DQNConfig()
    env_cfg = ENVConfig()

    # # # train
    # env, agent = env_agent_config(env_cfg, agent_cfg, seed=1)
    # rewards, ma_rewards, loss = train(agent_cfg, env, agent, is_display=False)
    # make_dir(agent_cfg.result_path, agent_cfg.model_path)
    # agent.save(path=agent_cfg.model_path)
    # # save_results(rewards, ma_rewards, tag='train', path=agent_cfg.result_path)
    # # plot_rewards(rewards, ma_rewards, tag="train",
    # #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)
    # # plot_losses(loss, algo=agent_cfg.algo, path=agent_cfg.result_path)
    # # # eval
    # env, agent = env_agent_config(env_cfg, agent_cfg, seed=10)
    # agent.load(path=agent_cfg.model_path)
    # rewards, ma_rewards = eval(agent_cfg, env, agent)
    # save_results(rewards, ma_rewards, tag='eval', path=agent_cfg.result_path)
    # plot_rewards(rewards, ma_rewards, tag="eval",
    #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)

    # baseline simple
    # env = DGEnv(env_cfg, deflection_angle=0)
    # rewards, ma_rewards, game_results = simple_tset(agent_cfg, env, is_display=True)
    # save_results(rewards, ma_rewards, tag='baseline', path=agent_cfg.result_path)
    # plot_rewards(rewards, ma_rewards, tag="baseline",
    #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)
    # diff_delta = []
    from sklearn.metrics import mean_squared_error

    mse = []
    target = 5 * np.ones(shape=(40, 1))
    for idx, spd in enumerate(range(10, 51, 10)):
        env_cfg.dog_spd = spd
        env, agent = env_agent_config(env_cfg, agent_cfg, seed=1)
        _, _, _, _ = train(agent_cfg, env, agent, is_display=False)
        make_dir(agent_cfg.result_path, agent_cfg.model_path)
        agent.save(path=agent_cfg.model_path)

        env, agent = env_agent_config(env_cfg, agent_cfg, seed=10)
        agent.load(path=agent_cfg.model_path)
        rewards, ma_rewards, _ = eval(agent_cfg, env, agent)
        save_results(rewards, ma_rewards, tag='eval', path=agent_cfg.result_path)
        # plot_rewards(rewards, ma_rewards, tag="eval",
        #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)
        pred = np.array(rewards[10:])
        mse.append(mean_squared_error(pred, target))
    print(mse)

    from sklearn.metrics import mean_squared_error

    mse = []
    target = 5 * np.ones(shape=(40, 1))
    train_times = []
    actionss = []
    for idx, spd in enumerate(range(10, 51, 10)):
        env_cfg.dog_spd = spd
        env, agent = env_agent_config(env_cfg, agent_cfg, seed=1)
        _, _, _, train_time = train(agent_cfg, env, agent, is_display=False)
        make_dir(agent_cfg.result_path, agent_cfg.model_path)
        agent.save(path=agent_cfg.model_path)
        #
        env, agent = env_agent_config(env_cfg, agent_cfg, seed=10)
        agent.load(path=agent_cfg.model_path)
        rewards, ma_rewards, actions = eval(agent_cfg, env, agent)
        save_results(rewards, ma_rewards, tag='eval', path=agent_cfg.result_path)
        # plot_rewards(rewards, ma_rewards, tag="eval",
        #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)
        # pred = np.array(rewards[10:])
        # mse.append(mean_squared_error(pred, target))
        # train_times.append(train_time)
        actions = abs(actions.count(0) - actions.count(1)) / len(actions)
        actionss.append(actions)

    np.save('./actions.npy', actionss)
    print(actionss)

    # print(train_times)
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # import seaborn
    # import seaborn as sns
    #
    # sns.set(style="darkgrid")
    # #
    # fix, ax = plt.subplots()
    # # delta = np.load('diff_delta.npy')
    # # nmse = [delta[i] * mse[i] for i in range(5)]
    # # #mse = delta * mse
    # ax.plot([1, 2, 3, 4, 5], train_times, label="train time")
    # # ax.plot([1, 2, 3, 4, 5], mse, label="mean squared error")
    # # ax.plot([1, 2, 3, 4, 5], nmse, label="tune mean squared error")
    # ax.set_xlabel('dog/goat speed rate')
    # ax.set_ylabel('train time')
    # ax.set_title("train time")
    # # ax.xlim(0, 4)
    # # ax.ylim(0, 1.)
    # ax.legend()
    # plt.show()

    # print(diff_delta)
    # np.save('./diff_delta.npy', np.array(diff_delta))
