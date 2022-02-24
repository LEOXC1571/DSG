#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :env.py
# @Time      :2021/8/6 下午3:34
# @Author    :miracleyin
# @Email     :miracleyin@live.com

import logging
import math
import time
import torch

import gym
from gym import spaces
from gym.utils import seeding
from utils import eucliDist, polar2cartesian, angle_normalize
import numpy as np
import random


class DGEnv(gym.Env):
    """
    Description:
        羊在半径为R的圆形圈内具有定常数率v和满足以下限制的任意转弯能力：逃逸路径
        上每一点与圆心的距离随时间单调不减。羊逃出圆形圈则胜。犬沿着圆周以定常速率V
        围堵以防止羊逃逸，任何时刻具有选择圆周的两个方向之一的能力

    Source:
        深圳杯 2021 D
    Enverment:
        Game area 500 * 500
        Cycle radius 200
        Goat speed
        Dog speed

    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Goat Position_X,Y        圆内
        1      Goat angel               0 2*pi
        2      Goat choice              -pi/2  pi/2
        3      Dog Position_X, Y        圆上
        4      Dog angel                0 2*pi
        5      Dog choice               0, 1

    Actions:
        Type: Discrete(3)
        Num    Action
        theta  Goat escape direction (-pi/2, pi/2)

        Note: 狗选择顺时针逆时针，羊选择当前朝向的偏转角度

    Reward:
         distence goat and center
         bigger than r
         distence dog and d
         bigger than size_d+size_g

    Starting State:
         goat: 位置x,y 初始方向

    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    # initialize the settings
    def __init__(self, cfg, deflection_angle=0.5 * math.pi, is_continue=False):  # 初始化环境
        # 固定游戏区域
        self.game_size = 500
        self.center_x = 250
        self.center_y = 250
        self.radius = 200
        self.end_step=[]
        self.og_step=0

        self.dog_spd = cfg.dog_spd
        self.dog_siz = cfg.dog_siz
        self.goat_spd = cfg.goat_spd  #
        self.goat_siz = cfg.goat_siz

        self.deflection_angle = deflection_angle  # action turn angle

        if is_continue:
            self.action_space = spaces.Box(low=0, high=math.pi, shape=(1,))
        else:
            self.action_space = spaces.Discrete(2)
        high = np.array([
            self.game_size,  # x`
            self.game_size,  # y
            2 * math.pi,
            self.game_size,
            self.game_size,
            2 * math.pi, ])
        low = np.array([
            0,
            0,
            0,
            0,
            0,
            0
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.seed()
        self.step_not_got_goat = None
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        return [seed]

    def step(self, action):
        self.og_step += 1

        if self.state is None:
            return None
        else:
            dog_x, dog_y, dog_angle, goat_x, goat_y, goat_angle = self.state

        dog_angle = (dog_angle ) % (2 * np.pi)
        goat_angle = (goat_angle ) % (2 * np.pi)
        dog_initx = dog_x
        dog_inity = dog_y
        goat_initx = goat_x
        goat_inity = goat_y

        dog_next_decision = self._dog_pursuit_policy(dog_angle, goat_angle)  # 实际上就是往劣弧追
        dog_angle_spd = self.dog_spd / self.radius  # 角速度转为线速度

        if dog_next_decision == 0:  #  逆时针
            dog_next_angle = dog_angle + dog_angle_spd
        elif dog_next_decision == 1:  # 顺时针
            dog_next_angle = dog_angle - dog_angle_spd
        else:
            dog_next_angle = dog_angle

        dog_next_x, dog_next_y = polar2cartesian((self.center_x, self.center_y), self.radius, dog_next_angle)

        if type(action) is type(1):  # 离散 DQN
            diff_center_goat_old = eucliDist((self.center_x, self.center_y),
                                             (goat_x, goat_y))
            deflection_angle = abs(math.cos(0.5 * (dog_angle - goat_angle))) * self.deflection_angle * (
                        1 - diff_center_goat_old / self.radius)  # 越接近边缘角度越小
            # deflection_angle = 0.5*self.deflection_angle
            if action == 0:  # 未修正的角度
                goat_next_temp_angle = goat_angle - deflection_angle
            else:
                goat_next_temp_angle = goat_angle + deflection_angle
        else:  # 连续 DDPG
            deflection_angle = angle_normalize(action)
            goat_next_temp_angle = goat_angle + deflection_angle
        goat_next_x, goat_next_y = polar2cartesian((goat_x, goat_y), self.goat_spd, goat_next_temp_angle)
        tune_angle = self._goat_face_tune(deflection_angle,
                                          (goat_x, goat_y),
                                          (goat_next_x, goat_next_y))
        goat_next_angle = goat_angle + tune_angle
        self.state = (dog_next_x, dog_next_y, dog_next_angle, goat_next_x, goat_next_y, goat_next_angle)
        diff_center_goat = eucliDist((self.center_x, self.center_y),
                                     (goat_next_x, goat_next_y))
        diff_dog_goat = eucliDist((dog_next_x, dog_next_y),
                                  (goat_next_x, goat_next_y))

        if (diff_center_goat > self.radius) and (diff_dog_goat > (self.goat_siz + self.dog_siz)):  #
            # 如果羊逃出，且没有被抓住 goat win
            done = True
            winner = 1
        elif diff_dog_goat <= (self.goat_siz + self.dog_siz):
            # 如果羊被抓住 dog win
            done = True
            winner = 0
        else:
            # 没被抓住 也没逃出去
            done = False
            winner = 0

        def ts_reward():

            if not done:  # 游戏没有结束 done = False
                if self.step_not_got_goat is None:  # 如果是第一步
                    self.step_not_got_goat = 1  # 初始化步
                    reward = 0
                else:
                    self.step_not_got_goat += 1  # 如果不是第一步
                    g_initdist = eucliDist((goat_initx, goat_inity), (self.center_x, self.center_y))
                    x = self.og_step - ((self.radius - g_initdist) / self.goat_spd)
                    # print(self.radius)
                    # print(self.goat_spd)
                    # print(g_initdist)
                    # print(self.og_step)
                    # print(self.radius - g_initdist / self.goat_spd)
                    # print(x)
                    reward = to_strategy(x , self.dog_spd / self.goat_spd)/15
                    # reward=0


            else:  # done = True

                if winner == 1:  # 胜利者是羊
                    reward = 10
                    self.end_step.append(self.step_not_got_goat),
                else:  # winner 1 胜利者是狗
                    reward = -10
                    self.end_step.append(self.step_not_got_goat)
                    ''' / self.step_not_got_goat'''

            return reward,self.end_step

        '''if not done:  # 游戏没有结束 done = False
            if self.step_not_got_goat is None:  # 如果是第一步
                self.step_not_got_goat = 1  # 初始化步
                reward = 0
            else:
                self.step_not_got_goat += 1  # 如果不是第一步
                reward = 5 / self.step_not_got_goat  #
        else:  # done = True
            if winner == 0:  # 胜利者是羊
                reward = 5
            else:  # winner 1 胜利者是狗
                reward = -5 / self.step_not_got_goat'''

        return np.array(self.state), ts_reward(), done, winner

    def _dog_pursuit_policy(self, dog_angle, goat_angle) -> int:

        goat_dog_diff = goat_angle - dog_angle
        if (0.05< goat_dog_diff <= math.pi) or ((-2 * math.pi + 0.05)< goat_dog_diff <= -math.pi):
            return 0
        elif (-math.pi < goat_dog_diff< -0.05 ) or (math.pi < goat_dog_diff < (2 * math.pi-0.05)):
            return 1
        else:
            return 2

    def _goat_face_tune(self, deflection, old_position, new_position, ):
        diff_pos = eucliDist(old_position, new_position)
        temp = diff_pos * math.sin(deflection) / (diff_pos * math.cos(deflection) + self.radius)
        return math.atan(temp)

    def goat_next_angle(self, deflection, old_position):
        r=eucliDist(old_position,(self.center_x,self.center_y))
        x=self.goat_spd
        goat_polar_angle_diff=math.atan((x*math.sin(deflection))/(x*math.cos(deflection)+r))
        return goat_polar_angle_diff

    def reset(self):
        # 初始化狗的位置，狗在圆上随机一点上出现
        goat_x, goat_y, goat_angle = self._init_goat(self.center_x,
                                                     self.center_y)  # 初始化羊的位置，羊在圆内随机一点上出现
        # angle是为了方便计算
        dog_x, dog_y, dog_angle = self._init_dog(self.center_x,
                                                 self.center_y)  # 根据羊的位置，初始化狼的决策
        self.og_step = 0
        self.state = (dog_x, dog_y, dog_angle, goat_x, goat_y, goat_angle)
        return np.array(self.state),self.og_step

    def _init_dog(self, x, y):
        r = self.radius
        alpha = random.uniform(0, 2 * math.pi)  # pi=0 位于圆的最右
        pos_x, pos_y = polar2cartesian((x, y), r, alpha)
        return pos_x, pos_y, alpha

    def _init_goat(self, x, y):
        r = self.radius
        p = random.uniform(0, r)
        alpha = random.uniform(0, 2 * math.pi)
        # beta = random.uniform(-math.pi / 2, math.pi / 2)
        pos_x, pos_y = polar2cartesian((x, y), p, alpha)
        return pos_x, pos_y, alpha

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.state is None:
            return None

        else:
            dog_x, dog_y, dog_angle, goat_x, goat_y, goat_angle = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.game_size, self.game_size)
            # 狗运动的圈
            self.cycle = rendering.make_circle(self.radius, filled=False)
            self.circletrans = rendering.Transform(translation=(self.center_x, self.center_y))
            self.cycle.add_attr(self.circletrans)
            self.cycle.set_color(1, 0, 0)

            # 狗
            self.dog = rendering.make_circle(self.dog_siz)
            self.dogtrans = rendering.Transform(translation=(dog_x, dog_y))
            self.dog.add_attr(self.dogtrans)
            self.dog.set_color(0, 0, 1)

            # 羊
            self.goat = rendering.make_circle(self.goat_siz)
            self.goattrans = rendering.Transform(translation=(goat_x, goat_y))
            self.goat.add_attr(self.goattrans)
            self.goat.set_color(0, 1, 0)

            self.viewer.add_geom(self.cycle)
            self.viewer.add_geom(self.goat)
            self.viewer.add_geom(self.dog)

        # 更新坐标
        self.dogtrans.set_translation(dog_x, dog_y)
        self.goattrans.set_translation(goat_x, goat_y)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class NormalizedActions(gym.ActionWrapper):
    ''' 将action范围重定在[0.1]之间
    '''

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action


class OUNoise(object):
    '''Ornstein–Uhlenbeck
    '''

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu

    def evolve_obs(self):
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()/15
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        get_action_return=np.clip(action + ou_obs, self.low, self.high)
        return np.clip(action + ou_obs, self.low, self.high)


def to_strategy(x, y):
    if x > 1:
        return -math.log(x, y)
    else:
        return 0


if __name__ == "__main__":
    pass
