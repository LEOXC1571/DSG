import math
import sys, os
import time
import numpy as np
import pandas as pd

from RL.env import DGEnv
import gym
import torch
import datetime
from utils import save_results, make_dir, save_esc_results
from utils import plot_rewards, plot_losses, plot_esccount
from agent import DQN
import matplotlib.pyplot as plt
from PIL import Image
import io

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
        self.eval_eps = 50
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
    def __init__(self, speed_ratio):
        self.dog_spd = 16
        self.dog_siz = 15
        self.goat_spd = 16 / speed_ratio
        self.goat_siz = 15


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
        state_save = np.zeros(shape=(1, 6))
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
            state_save = np.vstack((state_save, next_state))
            state = next_state
            loss = agent.update()
            if loss != 0:
                total_loss += loss.cpu().detach().numpy()
            if done:
                env.close()
                break
        train_time += time.time() - t1
        if i_ep < 10:
            trj_both = state_save
            trj_both = trj_both[1:]
            trj_plot(trj_both, i_ep)
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
    # trj_both = []
    # trj_both = np.array(trj_both)
    for i_ep in range(agent_cfg.eval_eps):
        ep_reward = 0
        state = env.reset()[0]
        og_step = env.reset()[1]
        # state_save = []
        state_save = np.zeros(shape=(1,6))
        while True:
            if is_display:
                env.render()
            action = agent.predict(state)
            actions.append(action)
            next_state, reward, done, winner = env.step(action)
            state_save = np.vstack((state_save, next_state))
            state = next_state
            ep_reward += reward[0]
            total_esc += winner
            if done:
                break
        rewards.append(ep_reward)
        if i_ep > agent_cfg.eval_eps-6:
            trj_both = state_save
            trj_both = trj_both[1:]
            # trj_plot(trj_both, i_ep)
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
    print('Start tp BASELINE SIMPLE!')
    print(f'Env:{cfg.env}, Algorithm: SIMPLE, Device:{cfg.device}')
    rewards = []
    ma_rewards = []
    game_results = []
    total_esc = 0
    esc_count = []
    for i_ep in range(cfg.eval_eps):
        state = env.reset()[0]
        og_step = env.reset()[1]
        state_save = np.zeros(shape=(1, 6))
        done = False
        ep_reward = 0
        i_step = 0
        esc_count.append(total_esc)
        while not done:
            i_step += 1
            action = env.action_space.sample()
            next_state, reward, done, winner = env.step(action)
            state_save = np.vstack((state_save, next_state))
            total_esc += winner
            if is_display:
                env.render()
            ep_reward += reward[0]
            state = next_state
        game_results.append(winner)
        rewards.append(ep_reward)
        if i_ep == agent_cfg.eval_eps-1:
            trj_both = state_save
            trj_both = trj_both[1:]
            # trj_plot(trj_both)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

    # for i in range(-1, -len(reward[1]), -1):
    #     reward[1][i] = reward[1][i] - reward[1][i - 1]
    # print(f'Average terminating step: {sum(reward[1]) / len(reward[1])}')
    print(f'Complete Eval! Escape rate is {total_esc / i_ep}')
    # print(esc_count)
    return rewards, ma_rewards, game_results, esc_count

def trj_plot(trj_both, ep):
    circle_x = []
    circle_y = []

    for i in range (1440):
        x = 250+200*math.cos(i/4)
        y = 250+200*math.sin(i/4)
        circle_x.append(x)
        circle_y.append(y)
    circle_x = np.array(circle_x)
    circle_y = np.array(circle_y)
    dog_x = trj_both[:,0]
    dog_y = trj_both[:,1]
    goat_x = trj_both[:,3]
    goat_y = trj_both[:,4]


    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    # set figure information
    # ax.set_title("Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0,500)
    ax.set_ylim(0,500)

    # ax.set_zlabel("z")

    # draw the figure, the color is r = read
    figure3 = ax.plot(circle_x, circle_y, c='grey',linewidth=0.1,linestyle=':')
    figure1 = ax.plot(dog_x, dog_y, c='r',linewidth=2,linestyle='-')
    figure2 = ax.plot(goat_x, goat_y, c='b',linewidth=2,linestyle='-')
    goatborn = ax.scatter(goat_x[0], goat_y[0], c='b')
    dog_arrow = ax.arrow(dog_x[-3], dog_y[-3], dog_x[-1]-dog_x[-3], dog_y[-1]-dog_y[-3], head_width=10, color='r')
    goat_arrow = ax.arrow(goat_x[-3], goat_y[-3], goat_x[-1] - goat_x[-3], goat_y[-1] - goat_y[-3], head_width=10, color='b')
    ax.axis('off')
    ax.legend([figure1, figure2], ['Dog trajectory', 'Sheep trajectory'], loc=2, bbox_to_anchor=(10, 10), borderaxespad=0.,
              fontsize='small', frameon=False, prop={'family': 'Times New Roman', 'size': 14})

    plt.show()
    png1 = io.BytesIO()
    fig.savefig(png1, format="png")
    png2 = Image.open(png1)
    png2.save("drl_trj{e}.tiff".format(e=ep))
    png1.close()

if __name__ == "__main__":
    ave_esc = pd.DataFrame()
    ave_base_esc = pd.DataFrame()
    # for sr in np.arange(1.5,math.pi+1,0.5):
    agent_cfg = DQNConfig()
    env_cfg = ENVConfig(speed_ratio=2)
    eval_esc_rate = []
    base_esc_rate = []
    episode = 2
    for i in range(1, episode):
        # train
        train_rewards = []
        env, agent = env_agent_config(env_cfg, agent_cfg, seed=i + 5)
        rewards, ma_rewards, loss, winner, train_esc_count = train(agent_cfg, env, agent, is_display=False)
        train_rewards.append(rewards)
        make_dir(agent_cfg.result_path, agent_cfg.model_path)
        agent.save(path=agent_cfg.model_path)
        save_results(rewards, ma_rewards, train_esc_count, tag='train', path=agent_cfg.result_path)
        # plot_rewards(rewards, ma_rewards, tag='train', algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)

        # eval
        eval_rewards = []
        env, agent = env_agent_config(env_cfg, agent_cfg, seed=i + episode)
        agent.load(path=agent_cfg.model_path)
        rewards, ma_rewards, _, eval_esc_count = eval(agent_cfg, env, agent, is_display=False)
        eval_rewards.append(rewards)
        save_results(rewards, ma_rewards, eval_esc_count, tag='eval', path=agent_cfg.result_path)
        eval_esc_pro = eval_esc_count[-1] / agent_cfg.eval_eps
        # plot_rewards(rewards, ma_rewards, tag='eval', algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)

        # baseline
        base_rewards = []
        env = DGEnv(env_cfg, deflection_angle=0)
        rewards, ma_rewards, game_results, base_esc_count = simple_test(agent_cfg, env, is_display=False)
        base_rewards.append(rewards)
        save_results(rewards, ma_rewards, base_esc_count, tag='baseline', path=agent_cfg.result_path)
        # plot_rewards(rewards, ma_rewards, tag="baseline",
        #              algo=agent_cfg.algo, env=agent_cfg.env, path=agent_cfg.result_path)
        diff_delta = []
        base_esc_pro = base_esc_count[-1] / agent_cfg.eval_eps
        eval_esc_rate.append(eval_esc_pro)
        base_esc_rate.append(base_esc_pro)
    eval_esc_rate = np.array(eval_esc_rate)
    base_esc_rate = np.array(base_esc_rate)
    print(np.mean(base_esc_rate))
    save_esc_results(eval_esc_rate, base_esc_rate, tag='DQN', path=agent_cfg.result_path)
    # df_temp = pd.DataFrame([[sr, np.mean(eval_esc_rate), np.mean(base_esc_rate)]],
    #                        columns=['sr', 'eval_esc', 'base_esc'])
    # ave_esc = ave_esc.append(df_temp)
# print(ave_esc)
# ave_esc.to_csv(agent_cfg.result_path + 'ave_esc.csv')





# plot_esccount(eval_esc_count, base_esc_count, tag='baseline', path=agent_cfg.result_path)
# train_rewards=pd.DataFrame(train_rewards)
# eval_rewards = pd.DataFrame(eval_rewards)
# base_rewards = pd.DataFrame(base_rewards)
# train_rewards_mean=train_rewards.mean(axis=1)
# eval_rewards_mean=eval_rewards.mean(axis=1)
# base_rewards_mean=base_rewards.mean(axis=1)
# print(train_rewards_mean)
# print(eval_rewards_mean)
# print(base_rewards_mean)
# print(train_esc_count[-1])
# print(eval_esc_count[-1])
# print(base_esc_count[-1])
