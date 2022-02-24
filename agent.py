#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :agent.py
# @Time      :2021/8/6 下午3:34
# @Author    :miracleyin
# @Email     :miracleyin@live.com
import random

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from models import MLP
from models import Actor, Critic, ReplayBuffer


class SimpleAgent(object):
    pass


class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon = 0
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(
            lambda: np.zeros(action_dim))  # A nested dictionary that maps state -> (action -> action-value)

    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.sample_count / self.epsilon_decay)
        # e-greedy policy
        if np.random.uniform(0, 1) > self.epsilon:
            action = self.predict(state)
        else:
            action = np.random.choice(self.action_dim)
        return action

    def predict(self, state):
        action = np.argmax(self.Q_table[str(state)])
        return action

    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward  # terminal state
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)

    def save(self, path):
        import dill
        torch.save(
            obj=self.Q_table,
            f=path + "Qleaning_model.pkl",
            pickle_module=dill
        )

    def load(self, path):
        import dill
        self.Q_table = torch.load(f=path + 'Qleaning_model.pkl', pickle_module=dill)


class DQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) #
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state):

        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            action = self.predict(state)
        else:
            action = random.randrange(self.action_dim)
        return action

    def predict(self, state):
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
        return action

    def update(self):

        if len(self.memory) < self.batch_size:
            return 0
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1)  # 例如tensor([[1],...,[0]])
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device)

        # 计算当前(s_t,a)对应的Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        # 计算所有next states的V(s_{t+1})，即通过target_net中选取reward最大的对应states
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # 计算 expected_q_value
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        # self.loss = F.smooth_l1_loss(q_values,expected_q_values.unsqueeze(1)) # 计算 Huber loss
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算 均方误差loss
        # 优化模型
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        loss.backward()
        # for param in self.policy_net.parameters():  # clip防止梯度爆炸
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # 更新模型
        return loss

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


class DDPG:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = cfg.device
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)

        # copy parameters to target net
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)  # 优化 critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0, 0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # convert variables to Tensor
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))
