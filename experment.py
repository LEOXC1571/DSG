#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :experment.py
# @Time      :2021/9/9 下午2:20
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sns

sns.set(style="darkgrid")

data = np.array([295, 371, 589, 613, 322])
delta = np.load('diff_delta.npy')
fix, ax = plt.subplots()

ax.plot([1, 2, 3, 4, 5], data, label="userful decision")
ax.set_xlabel('dog/goat speed rate')
ax.set_ylabel('number of decision')
ax.set_title("userful decision")
# ax.xlim(0, 4)
# ax.ylim(0, 1.)
ax.legend()
plt.show()
# x = np.linspace(0, 2, 100)
#
# fig, ax = plt.subplots()
# ax.plot(x, x, label='linear')
# ax.plot(x, x**2, label='quadratic')
# ax.plot(x, x**3, label='cubic')
# ax.set_xlabel('x label')
# ax.set_ylabel('y label')
# ax.set_title("Simple Plot")
# ax.legend()
# plt.show()
# df = pd.DataFrame(delta, columns={'difficulty delta'})
# fig, ax = plt.subplots(1, 1, figsize=(100, 50))
# sns.set(style="whitegrid")  #
# sns.relplot(kind="line", data=df)
# plt.legend(loc='upper right')
# plt.xlim(0, 4)
# plt.ylim(0, 1.)
# plt.xlabel("dog/goat speed")
# plt.ylabel("delta")
# ax.xaxis.set_ticks([1, 2, 3, 4, 5])
# plt.show()
