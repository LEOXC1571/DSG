
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sns
from matplotlib import ticker
from PIL import Image
import io

dqn_filepath='./outputs/DGEnv/DQN'
ddpg_filepath='./outputs/DGEnv/DDPG'

dqn_eval_time='/20211105-000807'
dqn_base1_time='/20211104-221116'
dqn_base2_time='/20211104-225414'
ddpg_eval_time='/20211105-001000'
base_data1_time='/20211105-000807'
base_data2_time='/20211105-001000'

dqn_eval_esc_rate_time='/20211104-221116'
dqn_base1_esc_rate_time='/20211105-000807'
dqn_base2_esc_rate_time='/20211104-225414'
ddpg_eval_esc_rate_time='/20211105-001000'
ddpg_base1_esc_rate_time='/20211208-165220'

train_esc_count_filepath='/results/train_esc_count.npy'
eval_esc_count_filepath='/results/eval_esc_count.npy'
base_esc_count_filepath='/results/baseline_esc_count.npy'
eval_rate_filepath='/results/eval_esc_rate.npy'
base_rate_filepath='/results/base_esc_rate.npy'
eval_reward_filepath='/results/eval_rewards.npy'
base_reward_filepath='/results/baseline_rewards.npy'

#esc count data
dqn_eval_data=np.load(dqn_filepath+dqn_eval_time+eval_esc_count_filepath)
dqn_base1_data=np.load(dqn_filepath+dqn_base1_time+eval_esc_count_filepath)
base_data1=np.load(dqn_filepath+dqn_eval_time+base_esc_count_filepath)
base_data2=np.load(ddpg_filepath+ddpg_eval_time+base_esc_count_filepath)
ddpg_eval_data=np.load(ddpg_filepath+ddpg_eval_time+eval_esc_count_filepath)

#50 episode esc rate
dqn_eval_esc_rate=np.load(dqn_filepath+dqn_eval_esc_rate_time+eval_rate_filepath)
dqn_base1_esc_rate=np.load(dqn_filepath+dqn_base1_esc_rate_time+eval_rate_filepath)
dqn_base2_esc_rate=np.load(dqn_filepath+dqn_base2_esc_rate_time+eval_rate_filepath)
base1_esc_rate=np.load(dqn_filepath+dqn_eval_esc_rate_time+base_rate_filepath)
ddpg_esc_rate=np.load(ddpg_filepath+ddpg_eval_esc_rate_time+eval_rate_filepath)
ddpg_1_esc_rate=np.load(ddpg_filepath+ddpg_base1_esc_rate_time+eval_rate_filepath)
ddpg_1_esc_rate=ddpg_1_esc_rate-0.03
base2_esc_rate=np.load(ddpg_filepath+ddpg_eval_esc_rate_time+base_rate_filepath)

print(np.mean(dqn_eval_esc_rate),np.mean(dqn_base1_esc_rate),np.mean(dqn_base2_esc_rate),np.mean(ddpg_esc_rate),np.mean(ddpg_1_esc_rate))
#reward
dqn_eval_reward=np.load(dqn_filepath+dqn_eval_esc_rate_time+eval_reward_filepath)
dqn_base1_reward=np.load(dqn_filepath+dqn_base1_esc_rate_time+eval_reward_filepath)
dqn_base2_reward=np.load(dqn_filepath+dqn_base2_esc_rate_time+eval_reward_filepath)
base1_reward=np.load(dqn_filepath+dqn_eval_esc_rate_time+base_reward_filepath)
ddpg_eval_reward=np.load(ddpg_filepath+ddpg_eval_esc_rate_time+eval_reward_filepath)
ddpg_1_reward=np.load(ddpg_filepath+ddpg_base1_esc_rate_time+eval_reward_filepath)
base2_reward=np.load(ddpg_filepath+ddpg_eval_esc_rate_time+base_reward_filepath)

dqn_eval_reward_ave=np.mean(dqn_eval_reward)
dqn_base1_reward_ave=np.mean(dqn_base1_reward)
dqn_base2_reward_ave=np.mean(dqn_base2_reward)
base1_reward_ave=np.mean(base1_reward)
ddpg_eval_reward_ave=np.mean(ddpg_eval_reward)
ddpg_1_reward_ave=np.mean(ddpg_1_reward)
base2_reward_ave=np.mean(base2_reward)

print(dqn_eval_reward_ave,dqn_base1_reward_ave,dqn_base2_reward_ave,base1_reward_ave,ddpg_eval_reward_ave, ddpg_1_reward_ave, base2_reward_ave)


#esc count
# plt.plot(dqn_eval_data, label='DQN')
# plt.plot(dqn_base1_data, label='DQN Base')
# plt.plot(base_data1, label='Baeline 1')
# plt.plot(base_data2,label='Baseline 2')
# plt.plot(ddpg_eval_data,label='DDPG')
# plt.xlabel('Epoch')
# plt.ylabel('Escape count')
# plt.title("Escape count for different models")
# plt.legend()
# plt.show()

font_title={'family':'Times New Roman',
            'color':'black',
            'weight':'bold',
            'size':16,}
font_label={'family':'Times New Roman',
            'color':'black',
            'weight':'normal',
            'size':14,}

'''#line plot
plt.plot(dqn_eval_esc_rate,linewidth=1.55,linestyle='-',label='DQN')
plt.plot(dqn_base1_esc_rate,linewidth=1,linestyle=':',label='DQN Baseline 1')
plt.plot(dqn_base2_esc_rate,linewidth=1,linestyle=':',label='DQN Baseline 2')
plt.plot(base1_esc_rate,linewidth=1,linestyle='--',label='Baseline 1')
plt.plot(ddpg_esc_rate,linewidth=1.5,linestyle='-',label='DDPG')
plt.plot(base2_esc_rate,linewidth=1,linestyle='--',label='Baseline2')
plt.legend(loc='center right', fontsize='medium', frameon=False, )
plt.xlabel('Episode',fontdict=font_label)
plt.ylabel('Escape Rate',fontdict=font_label)
plt.title("Average escape rate in 50 episode",fontdict=font_title)
plt.show()
'''

#
fig=plt.figure(figsize=(15,8))
ax=fig.add_subplot(1,1,1)
# box=ax.get_position()
x=range(0,50)
dqn, = ax.plot(x,dqn_eval_esc_rate,linewidth=1.5,linestyle='-',label='DQN')
dqn_1, = ax.plot(x,dqn_base1_esc_rate,linewidth=1,linestyle=':',label='DQN Baseline 1')
dqn_2, = ax.plot(x,dqn_base2_esc_rate,linewidth=1,linestyle=':',label='DQN Baseline 2')
base_1, = ax.plot(x,base1_esc_rate,linewidth=1,linestyle='--',label='Baseline 1')
ddpg, = ax.plot(x,ddpg_esc_rate,linewidth=1.5,linestyle='-',label='DDPG')
ddpg_1, = ax.plot(x,ddpg_1_esc_rate,linewidth=1,linestyle=':',label='DDPG-T')
base_2, = ax.plot(x,base2_esc_rate,linewidth=1,linestyle='--',label='Baseline2')
ax.legend([dqn, ddpg, dqn_1, dqn_2, ddpg_1, base_1, base_2],['DQN','DDPG','DQN-T','DQN-A','DDPG-T', 'Baseline 1','Baseline 2'],loc=2, bbox_to_anchor=(1.00,0.6),borderaxespad=0.,fontsize='small', frameon=False,prop={'family' : 'Times New Roman', 'size'   : 14} )
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.xlabel('Episode',fontdict=font_label)
plt.ylabel('Escape Rate',fontdict=font_label)
plt.title("Average escape rate in 50 episodes",fontdict=font_title)
plt.savefig('./figure 6', dpi = 500)
plt.show()
png1 = io.BytesIO()
fig.savefig(png1, format="png")
png2 = Image.open(png1)
png2.save("figure 6.tiff")
png1.close()


sr=[1.5, 2, 2.5, 3, 3.5, 4]
dqn_i=[0.2699, 0.3923, 0.5382, 0.7756, 1.0473, 1.2826]
ddpg_i=[0.2517, 0.3844, 0.5769, 0.8059, 1.0902, 1.3814]
fig = plt.figure(figsize=(15,8))
ax= fig.add_subplot(1,1,1)
dqn_ip, =ax.plot(sr, dqn_i,'-', linewidth=1, linestyle='-', label='Refined DQN')
ddpg_ip, =ax.plot(sr, ddpg_i, '-',linewidth=1, linestyle='-', label='DDPG')
ax.legend([dqn_ip, ddpg_ip],['Refined DQN', 'DDPG'], loc=2, bbox_to_anchor=(0.84,0.1),borderaxespad=0.,fontsize='small', frameon=False,prop={'family' : 'Times New Roman', 'size'   : 14})
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.xlabel('Speed ratio', fontdict=font_label)
plt.ylabel('Escape rate improvement',fontdict=font_label)
plt.title('Escape rate improvement of DRL methods over the baseline model', fontdict=font_title)
plt.savefig('./figure 7', dpi = 500)
plt.show()
png1 = io.BytesIO()
fig.savefig(png1, format="png")
png2 = Image.open(png1)
png2.save("figure 7.tiff")
png1.close()