#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2021/8/6 下午3:34
# @Author    :miracleyin
# @Email     :miracleyin@live.com

import torch
import numpy as np
import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#calculate the distance
def eucliDist(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)



#calculate the position change
def polar2cartesian(pos, p, angle):
    x, y = pos[0], pos[1]
    x = x + p * math.cos(angle)
    y = y + p * math.sin(angle)
    return x, y

#reward plot output
def plot_rewards(rewards, ma_rewards, tag="train", env='CartPole-v0', algo="DQN", save=True, path='./'):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo, env))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if save:
        plt.savefig(path + "{}_rewards_curve".format(tag))
    plt.show()

def plot_esccount(eval_esc_count, base_esc_count, tag="train", env='CartPole-v0', algo="DQN", save=True, path='./'):
    sns.set()
    plt.title("Escape Count of {} for {}".format(algo, env))
    plt.xlabel('epsiodes')
    plt.plot(eval_esc_count, label='DQN')
    plt.plot(base_esc_count, label='Baseline')
    plt.legend()
    if save:
        plt.savefig(path + "{}_esccount_curve".format(tag))
    plt.show()

#loss plot output
def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()

#save training/eval results
def  save_results(rewards, ma_rewards, esc_count, tag='train', path='./results'):
    '''save rewards and ma_rewards
    '''
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    np.save(path + '{}_esc_count.npy'.format(tag), esc_count)
    print('results saved!')

def  save_esc_results(eval_esc_rate, base_esc_rate, tag='eval', path='./results'):
    '''save rewards and ma_rewards
    '''
    np.save(path + 'eval_esc_rate.npy', eval_esc_rate)
    np.save(path + 'base_esc_rate.npy', base_esc_rate)
    # np.save(path + '{}_esc_count.npy'.format(tag), esc_count)
    print('results saved!')

def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

#limit the angel to [0,2pi)
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
