# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 18:18:24 2017

@author: hyj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 21:43:55 2017

@author: hyj
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
filepath=os.path.abspath('..')+"/bin"  #表示当前所处的文件夹上一级文件夹的绝对路径

position = []
quaterntions = []
timestamp = []
tx_index1 = 5
tx_index2 = 1
with open(filepath + '/imu_pose.txt', 'r') as f:  # imu_circle   imu_spline

    data = f.readlines()  # txt中所有字符串读入data
    for line in data:
        odom = line.split()  # 将单个数据分隔开存好
        numbers_float = map(float, odom)  # 转化为浮点数

        position.append([numbers_float[tx_index1], numbers_float[tx_index1 + 1], numbers_float[tx_index1 + 2]])

position1 = []
quaterntions1 = []
timestamp1 = []
with open(filepath + '/pose_output_with_noise.txt', 'r') as f:  # imu_pose   imu_spline

    data = f.readlines()  # txt中所有字符串读入data
    for line in data:
        odom = line.split()  # 将单个数据分隔开存好
        numbers_float = map(float, odom)  # 转化为浮点数
        position1.append([numbers_float[tx_index2], numbers_float[tx_index2 + 1], numbers_float[tx_index2 + 2]])



    ### plot 3d
fig = plt.figure()
ax = fig.gca(projection='3d')

xyz = zip(*position)
xyz1 = zip(*position1)

print
ax.plot(xyz[0], xyz[1], xyz[2], label='imu_pose')
ax.plot(xyz1[0], xyz1[1], xyz1[2], label='vins_pose')

ax.legend()

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
