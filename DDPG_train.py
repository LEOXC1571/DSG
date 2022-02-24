import sys, os
import time
import pandas as pd
import math
from DDPG_env import DGEnv

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from agent import DDPG
from DDPG_env import NormalizedActions, OUNoise
import gym
from utils import save_results, make_dir, plot_rewards, plot_esccount, save_esc_results
from DQN_train import trj_plot

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
        self.train_eps = 50
        self.eval_eps = 50
        self.eval_steps = 200
        self.target_update = 4
        self.hidden_dim = 30
        self.soft_tau = 1e-2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ENVConfig:
    def __init__(self, speed_ratio):
        self.dog_spd = 16
        self.dog_siz = 15
        self.goat_spd = 16 / speed_ratio
        self.goat_siz = 15


def env_agent_config(env_cfg, agent_cfg, seed=42):
    env = NormalizedActions(DGEnv(env_cfg, is_continue=True))
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim, action_dim, agent_cfg)
    return env, agent


def train(cfg, env, agent, is_display=True):
    print('Start to train ! ')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    ou_noise = OUNoise(env.action_space)
    rewards = []
    ma_rewards = []  # moving average rewards
    train_times = 0
    total_esc = 0
    esc_count = []
    for i_ep in range(cfg.train_eps):
        t1 = time.time()
        state = (env.reset())[0]
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0

        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, winner = env.step(action)
            if is_display:
                env.render()
            ep_reward += reward[0]
            total_esc += winner
            agent.memory.push(state, action, reward[0], next_state, done)
            agent.update()
            state = next_state

        if (i_ep + 1) % 10 == 0:
            print(f'Episode:{i_ep + 1}/{agent_cfg.train_eps}, Reward:{ep_reward},Capture Rate:{total_esc / (i_ep + 1)}')
            esc_count.append(total_esc)
        else:
            esc_count.append(total_esc)

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        train_times += time.time() - t1

    # for i in range(-1, -len(reward[1]), -1):
    #     reward[1][i] = reward[1][i] - reward[1][i - 1]
    # print(f'Average terminating step: {sum(reward[1]) / len(reward[1])}')
    print('Complete training！')
    return rewards, ma_rewards, train_times / 300, esc_count


def eval(cfg, env, agent, is_display=True):
    print('Start to Eval ! ')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    std_actions = 0
    actionss = []
    total_esc = 0
    esc_count = []
    for i_ep in range(cfg.eval_eps):
        state = (env.reset())[0]
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            actionss.append(action)
            next_state, reward, done, winner = env.step(action)
            if is_display:
                env.render()
            ep_reward += reward[0]
            total_esc += winner
            state = next_state
        if (i_ep + 1) % 10 == 0:
            print(f'Episode:{i_ep + 1}/{cfg.eval_eps}, Reward:{ep_reward},Capture Rate:{total_esc / (i_ep + 1)}')
        esc_count.append(total_esc)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        temp_action = np.array(actionss)
        std_actions += np.std(temp_action)
        actionss = []
    # for i in range(-1, -len(reward[1]), -1):
    #     reward[1][i] = reward[1][i] - reward[1][i - 1]
    # print(f'Average terminating step: {sum(reward[1]) / len(reward[1])}')
    print('Complete Eval！')
    return rewards, ma_rewards, std_actions, esc_count


def random_test(cfg, env, is_display=True):
    print('Start to BASELINE RANDOM! ')
    print(f'Env:{cfg.env}, Algorithm: RANDOM, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    gr = []
    total_esc = 0
    esc_count = []
    state_save = np.zeros(shape=(1, 6))
    for i_ep in range(cfg.eval_eps):
        state = (env.reset())[0]
        done = False
        ep_reward = 0
        i_step = 0
        esc_count.append(total_esc)
        while not done:
            i_step += 1
            action = env.action_space.sample()
            next_state, reward, done, winner = env.step(action)
            state_save = np.vstack((state_save, next_state))
            if is_display:
                env.render()
            ep_reward += reward[0]
            total_esc += winner
            state = next_state
        if i_ep < 10:
            trj_both = state_save
            trj_both = trj_both[1:]
            trj_plot(trj_both, i_ep)
        if (i_ep + 1) % 10 == 0:
            print(f'Episode:{i_ep + 1}/{cfg.eval_eps}, Reward:{ep_reward},Capture Rate:{total_esc / (i_ep + 1)}')
        rewards.append(ep_reward)
        gr.append(winner)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    # for i in range(-1, -len(reward[1]), -1):
    #     reward[1][i] = reward[1][i] - reward[1][i - 1]
    # print(f'Average terminating step: {sum(reward[1]) / len(reward[1])}')
    print(f'Complete Eval! Escape rate is {total_esc / i_ep}')
    print('Complete Eval！')
    return rewards, ma_rewards, gr, esc_count


if __name__ == "__main__":
    ave_esc = pd.DataFrame()
    ave_base_esc = pd.DataFrame()
    # for sr in np.arange(1.5, math.pi + 1, 0.5):
    agent_cfg = DDPGConfig()
    env_cfg = ENVConfig(speed_ratio=2)
    eval_esc_rate = []
    base_esc_rate = []
    episode = 2
    for i in range(1, episode):
        # train
        env, agent = env_agent_config(env_cfg, agent_cfg, seed=i + 2)
        rewards, ma_rewards, _, train_esc_count = train(agent_cfg, env, agent, is_display=False)
        make_dir(agent_cfg.result_path, agent_cfg.model_path)
        agent.save(path=agent_cfg.model_path)
        save_results(rewards, ma_rewards, train_esc_count, tag='train', path=agent_cfg.result_path)
        # plot_rewards(rewards, ma_rewards, tag="train",
        #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)

        # eval
        env, agent = env_agent_config(env_cfg, agent_cfg, seed=i + episode)
        agent.load(path=agent_cfg.model_path)
        rewards, ma_rewards, _, eval_esc_count = eval(agent_cfg, env, agent, is_display=False)
        save_results(rewards, ma_rewards, eval_esc_count, tag='eval', path=agent_cfg.result_path)
        # plot_rewards(rewards, ma_rewards, tag="eval",
        #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)
        eval_esc_pro = eval_esc_count[-1] / agent_cfg.eval_eps

        # baseline random
        env = DGEnv(env_cfg, is_continue=True)
        rewards, ma_rewards, _, base_esc_count = random_test(agent_cfg, env, is_display=False)
        save_results(rewards, ma_rewards, base_esc_count, tag='baseline', path=agent_cfg.result_path)
        # plot_rewards(rewards, ma_rewards, tag="baseline",
        #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)
        base_esc_pro = base_esc_count[-1] / agent_cfg.eval_eps
        # plot_esccount(eval_esc_count, base_esc_count, tag='baseline', path=agent_cfg.result_path)
        eval_esc_rate.append(eval_esc_pro)
        base_esc_rate.append(base_esc_pro)
    eval_esc_rate = np.array(eval_esc_rate)
    base_esc_rate = np.array(base_esc_rate)
    print(np.mean(base_esc_rate))
    save_esc_results(eval_esc_rate, base_esc_rate, tag='DDPG', path=agent_cfg.result_path)
    # df_temp = pd.DataFrame([[sr, np.mean(eval_esc_rate), np.mean(base_esc_rate)]],
    #                        columns=['sr', 'eval_esc', 'base_esc'])
    # ave_esc = ave_esc.append(df_temp)
# print(ave_esc)
# ave_esc.to_csv(agent_cfg.result_path + 'ave_esc.csv')
