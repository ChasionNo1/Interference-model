"""
模拟干扰模型
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Circle
import math
import random


class Simulation:
    def __init__(self, seed, shape, width, height):
        self.seed = seed
        self.shape = shape
        self.width = width
        self.height = height
        self.fig = plt.figure()

    def plot_inter(self):
        if self.shape == 'ellipse':
            a = self.width / 2
            b = self.height / 2
            ax = self.fig.add_subplot(111)
            ell = Ellipse(xy=(a+2.5, b-2.5), width=self.width, height=self.height, facecolor='none', edgecolor='black', linestyle='solid',
                          linewidth=2.0, angle=90, alpha=0.3)
            # 椭圆：x^2/a^2 + y^2/b^2 = 1

            np.random.seed(self.seed)
            # x的坐标是向右平移了a个单位
            point_x = 2*a*np.random.random(10)
            # 改为原点为圆心
            rec_point_x = [x - a for x in point_x]
            # 原点为圆心的y轴坐标
            point_y = [2 * math.sqrt(b ** 2*(1 - x**2/a**2)) for x in rec_point_x]
            # 在这个范围内生成随机数
            point_y_list = [x * random.random() for x in point_y]
            xy = zip(point_x, point_y_list)
            print(list(xy))
            plt.scatter(point_x, point_y_list)
            ax.add_patch(ell)
            plt.axis('equal')
            plt.show()


pic = Simulation(1, 'ellipse', 20, 15)
pic.plot_inter()



# fig = plt.figure()
# ax = fig.add_subplot(111)
# ell = Ellipse(xy=(0.0, 0.0), width=15, height=20, facecolor='none', edgecolor='black', linestyle='solid', linewidth=2.0, angle=90, alpha=0.3)
# ax.add_patch(ell)
# x, y = 0, 0
# ax.plot(x, y, 'ro')
# # plt.axis('scaled')
# plt.axis('equal')
# plt.show()
# # print(ell.get_patch_transform())



