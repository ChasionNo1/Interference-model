"""
模拟干扰模型
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Circle, Rectangle
import math


class Simulation:
    def __init__(self, seed, shape, width, height, offset):
        self.seed = seed
        self.shape = shape
        self.width = width
        self.height = height
        self.fig = plt.figure()
        self.offset = offset
        np.random.seed(seed)
        self.inter_position = []
        self.outer_position = []

    def plot_inter_and_outer_point(self):
        if self.shape == 'ellipse':
            a_offset = self.width / 2 + self.offset
            b_offset = self.height / 2 + self.offset
            a = self.width / 2
            b = self.height / 2
            ax = self.fig.add_subplot(111)
            ell_inter = Ellipse(xy=(a_offset, b_offset), width=self.width, height=self.height, facecolor='none', edgecolor='black', linestyle='solid',
                          linewidth=2.0, alpha=0.3)
            ell_outer = Ellipse(xy=(a_offset, b_offset), width=self.width + 2*self.offset, height=self.height + 2*self.offset, facecolor='none', edgecolor='red', linestyle='solid',
                          linewidth=2.0, alpha=0.3)
            # 椭圆：x^2/a^2 + y^2/b^2 = 1

            # x的坐标是向右平移了a个单位
            point_x = 2*a*np.random.random(20)
            # 改为原点为圆心
            # rec_point_x = [x - a for x in point_x]
            # 原点为圆心的y轴坐标
            delta = [math.sqrt(b ** 2*(1 - (x-a)**2/a**2)) for x in point_x]
            y_delta1 = [b + x for x in delta]
            y_delta2 = [b - x for x in delta]
            chazhi = list(map(lambda x: x[0]-x[1], zip(y_delta1, y_delta2)))
            # point_y = list(map(lambda x: y_delta2[0] + np.random.random()*chazhi[0], zip(y_delta2, chazhi)))
            point_y = []
            point_x = [x + 5 for x in point_x]
            for i in range(len(point_x)):
                temp = y_delta2[i] + chazhi[i] * np.random.random() + 5
                point_y.append(temp)

            """
            画外部干扰点 m
            """
            point_x_outer = 2*a_offset*np.random.random(20)
            point_y_outer = 2*b_offset*np.random.random(20)
            pm1 = []
            pm2 = []
            for i in range(len(point_y_outer)):
                m1 = (point_x_outer[i]-a_offset)**2/a_offset**2 + (point_y_outer[i] - b_offset)**2/b_offset**2
                m2 = (point_x_outer[i]-a_offset)**2/a**2 + (point_y_outer[i] - b_offset)**2/b**2
                if m1 < 1 and m2 > 1:
                    pm1.append(point_x_outer[i])
                    pm2.append(point_y_outer[i])

            plt.scatter(point_x, point_y)
            plt.scatter(pm1, pm2)
            self.inter_position = list(zip(point_x, point_y))
            self.outer_position = list(zip(pm1, pm2))
            ax.add_patch(ell_inter)
            ax.add_patch(ell_outer)
            plt.axis('equal')
            plt.show()
        if self.shape == 'rectangle':
            # 长方形的宽与高
            a = self.width
            b = self.height
            a_offset = a + 2*self.offset
            b_offset = b + 2*self.offset
            rec = Rectangle(xy=(0.0 + self.offset, 0.0 + self.offset), width=a, height=b, facecolor='none', edgecolor='black', linewidth=1, linestyle='solid')
            rec2 = Rectangle(xy=(0.0, 0.0), width=a_offset, height=b_offset, facecolor='none', edgecolor='red', linewidth=1, linestyle='solid')
            point_x = a * np.random.random(20) + self.offset
            point_y = b * np.random.random(20) + self.offset

            """
            画干扰顶点
            """
            point_x_outer = a_offset * np.random.random(20)
            point_y_outer = b_offset * np.random.random(20)
            pm1 = []
            pm2 = []
            for i in range(len(point_y_outer)):
                if self.offset < point_x_outer[i] < self.offset + a and self.offset < point_y_outer[i] < self.offset + b:
                    pass
                else:
                    pm1.append(point_x_outer[i])
                    pm2.append(point_y_outer[i])

            plt.scatter(point_x, point_y)
            plt.scatter(pm1, pm2)
            self.inter_position = list(zip(point_x, point_y))
            self.outer_position = list(zip(pm1, pm2))
            ax = self.fig.add_subplot(111)
            ax.add_patch(rec)
            ax.add_patch(rec2)
            plt.axis('equal')
            plt.show()

        if self.shape == 'circle':
            # 半径是self.width/2
            r = self.width/2
            r_offset = r + self.offset
            cir = Circle(xy=(r_offset, r_offset), radius=self.width/2, facecolor='none', edgecolor='black', linewidth=1, linestyle='solid')
            cir2 = Circle(xy=(r_offset, r_offset), radius=r_offset, facecolor='none', edgecolor='red', linewidth=1, linestyle='solid')
            point_x = self.width * np.random.random(20) + self.offset
            delta = [math.sqrt(r**2 - (x - r_offset) ** 2) for x in point_x]
            y_delta1 = [r + x for x in delta]
            y_delta2 = [r - x for x in delta]
            chazhi = list(map(lambda x: x[0] - x[1], zip(y_delta1, y_delta2)))
            # point_y = list(map(lambda x: y_delta2[0] + np.random.random()*chazhi[0], zip(y_delta2, chazhi)))
            point_y = []
            for i in range(len(point_x)):
                temp = y_delta2[i] + chazhi[i] * np.random.random() + self.offset
                point_y.append(temp)

            """
            生成干扰顶点
            """
            point_x_outer = 2 * r_offset * np.random.random(20)
            point_y_outer = 2 * r_offset * np.random.random(20)
            pm1 = []
            pm2 = []
            for i in range(len(point_y_outer)):
                m1 = (point_x_outer[i] - r_offset) ** 2 + (point_y_outer[i] - r_offset) ** 2
                m2 = (point_x_outer[i] - r_offset) ** 2 + (point_y_outer[i] - r_offset) ** 2
                if m1 < r_offset ** 2 and m2 > r**2:
                    pm1.append(point_x_outer[i])
                    pm2.append(point_y_outer[i])
            plt.scatter(point_x, point_y)
            plt.scatter(pm1, pm2)
            self.inter_position = list(zip(point_x, point_y))
            self.outer_position = list(zip(pm1, pm2))
            # ax = self.fig.add_subplot(111)
            # ax.add_patch(cir)
            # ax.add_patch(cir2)
            # plt.axis('equal')
            # plt.show()


if __name__ == '__main__':
    pic = Simulation(5, 'circle', 20, 16, 5)
    pic.plot_inter_and_outer_point()







