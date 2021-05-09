# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/26 19:41
# @Author    :   Chasion
# Description:   画图，战术通信网
import matplotlib.pyplot as plt
import numpy as np


def plot_result():
    """
    初步想法是：
    在矩形内生成一些随机点，10个
    然后在四个方向设置干扰机，三角形

    全都是黑白图
    :return:
    """
    x = np.random.randint(10, 40, size=(8, ))
    y = np.random.randint(15, 30, size=(8, ))
    trix = [5, 45, 5, 45]
    triy = [5, 5, 35, 35]
    plt.scatter(x, y, c='black', label='user', linewidths=2)
    plt.scatter(trix, triy, c='black', marker='^', label='jammer', linewidths=3.5)

    plt.xlim(0, 50)
    plt.ylim(0, 40)
    plt.title('Network Topology Simulation')
    plt.xlabel('x/km')
    plt.ylabel('y/km')
    plt.legend(loc=9)
    plt.show()


if __name__ == '__main__':
    seed = 25
    np.random.seed(seed)
    plot_result()
