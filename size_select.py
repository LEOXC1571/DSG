import sys, os
import time
import numpy as np
import pandas as pd
import math
from RL.base_env import DGEnv
import gym
import torch
import datetime
from utils import save_results, make_dir, save_esc_results
from utils import plot_rewards, plot_losses, plot_esccount
from agent import DQN


os.getcwd()
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")


class DQNConfig:
    def __init__(self):
        self.algo = "DQN"
        self.env = 'DGEnv'
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + self.algo + '/' + curr_time + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env + \
                          '/' + self.algo + '/' + curr_time + '/models/'

        # model set

        self.train_eps = 1500
        self.eval_eps = 1000
        self.gamma = 0.95
        self.epsilon_start = 0.90
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.001
        self.memory_capacity = 10000
        self.batch_size = 64
        self.target_update = 4
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = 256


class ENVConfig:
    def __init__(self, dog_spd=16, goat_spd=8, size=10):
        self.dog_spd = dog_spd
        self.dog_siz = 10
        self.goat_spd = goat_spd
        self.goat_siz = size


def env_agent_config(env_cfg, agent_cfg, seed=42):
    env = DGEnv(env_cfg, is_continue=False)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, action_dim, agent_cfg)
    return env, agent


def train(agent_cfg, env, agent, is_display=True):
    print('Start to train!')
    print(f'Env:{agent_cfg.env},Algorithm:{agent_cfg.algo},Device:{agent_cfg.device}')
    rewards = []
    ma_rewards = []
    loss_list = []
    train_time = 0
    total_esc = 0
    esc_count = []

    for i_ep in range(agent_cfg.train_eps):
        t1 = time.time()
        state = (env.reset())[0]
        og_step = (env.reset())[1]
        ep_reward = 0
        total_loss = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, winner = env.step(action)
            if is_display:
                env.render()
            ep_reward += reward[0]
            total_esc += winner
            # esc_count.append(total_esc)
            # print(esc_count)
            agent.memory.push(state, action, reward[0], next_state, done)
            state = next_state
            loss = agent.update()
            if loss != 0:
                total_loss += loss.cpu().detach().numpy()
            if done:
                env.close()
                break
        train_time += time.time() - t1

        if (i_ep + 1) % agent_cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (i_ep + 1) % 10 == 0:
            print(f'Episode:{i_ep + 1}/{agent_cfg.train_eps},Reward:{ep_reward},Escape Rate:{total_esc / (i_ep + 1)}')
            esc_count.append(total_esc)
        else:
            esc_count.append(total_esc)

        rewards.append(ep_reward)
        loss_list.append(total_loss)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0 / 1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

    # for i in range(-1, -len(reward[1]), -1):
    #     reward[1][i] = reward[1][i] - reward[1][i - 1]
    # for i in reward[1][::-1]:
    # print(og_step)
    # print(reward[1])
    # print(f'Average terminating step: {sum(reward[1]) / len(reward[1])}')

    print('Complete training!')
    # print(esc_count)
    return rewards, ma_rewards, loss_list, train_time / 1000, esc_count


def eval(agent_cfg, env, agent, is_display=True):
    print('Start to eval!')
    print(f'Env{agent_cfg.env},Algorithm:{agent_cfg.algo},Device:{agent_cfg.device}')
    rewards = []
    ma_rewards = []
    actions = []
    total_esc = 0
    esc_count = []
    for i_ep in range(agent_cfg.eval_eps):
        ep_reward = 0
        state = env.reset()[0]
        og_step = env.reset()[1]
        while True:
            if is_display:
                env.render()
            action = agent.predict(state)
            actions.append(action)
            next_state, reward, done, winner = env.step(action)
            state = next_state
            ep_reward += reward[0]
            total_esc += winner
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 5 == 0:
            print(
                f'Episode:{i_ep + 1}/{agent_cfg.eval_eps},reward:{ep_reward:.1f}, Escape rate:{total_esc / (i_ep + 1)}')
            esc_count.append(total_esc)
        else:
            esc_count.append(total_esc)

    # for i in range(-1, -len(reward[1]), -1):
    #     reward[1][i] = reward[1][i] - reward[1][i - 1]
    # print(f'Average terminating step: {sum(reward[1]) / len(reward[1])}')

    print('Complete evaling!')
    return rewards, ma_rewards, actions, esc_count


def simple_test(cfg, env, is_display=True):
    # print('Start tp BASELINE SIMPLE!')
    # print(f'Env:{cfg.env}, Algorithm: SIMPLE, Device:{cfg.device}')
    rewards = []
    ma_rewards = []
    game_results = []
    total_esc = 0
    esc_count = []
    for i_ep in range(cfg.eval_eps):
        state = env.reset()[0]
        og_step = env.reset()[1]
        done = False
        i_step = 0
        esc_count.append(total_esc)
        while not done:
            i_step += 1
            action = env.action_space.sample()
            next_state, done, winner = env.step(action)
            total_esc += winner
            if is_display:
                env.render()
            state = next_state
        game_results.append(winner)

    # print(f'Complete Eval! Escape rate is {total_esc / i_ep}')
    return rewards, ma_rewards, game_results, esc_count


def theo_esc_rate(sr):
    temp=0.5
    if sr <= math.pi:
        temp = sr / (3 * math.pi)
    elif sr > math.pi:
        temp = sr / (3 * math.pi) * (1 - (1 - math.pi / sr) ** 3)
    esc_rate=1-temp
    return esc_rate


if __name__ == "__main__":
    agent_cfg = DQNConfig()
    base_esc_rate = pd.DataFrame()
    pd_esc_rate=pd.DataFrame()
    pd_siz=pd.DataFrame()
    for sr in np.arange(1, math.pi + 1, 0.2):
        theo_er = theo_esc_rate(sr)
        # print(theo_er)
        temp_esc = pd.DataFrame()
        for j in np.arange(1, 40):
            gs = 30 / sr
            env_cfg = ENVConfig(dog_spd=30, goat_spd=gs, size=j)
            env = DGEnv(env_cfg, deflection_angle=0)
            _, _, _, base_esc_count = simple_test(agent_cfg, env, is_display=False)
            base_esc_pro = pd.DataFrame([[j,theo_er, base_esc_count[-1] / agent_cfg.eval_eps,abs(theo_er-base_esc_count[-1] / agent_cfg.eval_eps)]],columns=['siz','tg_rate','esc_rate','diff'],index=[j])
            temp_esc = temp_esc.append(base_esc_pro)
            # save_esc_results(eval_esc_rate,base_esc_rate,tag='DQN', path=agent_cfg.result_path)
            # plot_esccount(eval_esc_count, base_esc_count, tag='baseline', path=agent_cfg.result_path)
        temp_esc=temp_esc.reset_index()
        temp_idx=temp_esc['diff'].argmin()
        pd_siz=pd_siz.append(temp_esc.loc[temp_idx])
        print(pd_siz)
        pd_sr_siz=pd.DataFrame([[sr,temp_idx+1]],columns=['speed ratio','ideal size'])
        base_esc_rate=base_esc_rate.append(pd_sr_siz)
    print(base_esc_rate)
    base_esc_rate.to_csv('sr_size.csv')
