"""
通信系统模型
data:2021/4/9
"""
import math
import numpy as np
from Data.simulation import Simulation
import random
import torch


BACKGROUND_NOISE = math.pow(10, -12)
SEND_POWER_MAX = 50
SEND_POWER_MIN = 5
INTERFERENCE_MAX = 1000
INTERFERENCE_MIN = 800
C = 3.0 * math.pow(10, 8)


class Receiver:
    def __init__(self, send_power, distance, f, n):
        """
        接收机的模型:直射波模型
        :param send_power: 发送功率
        :param distance: 距离，短波通信
        :param f: 使用频率的中频
        :param n: 衰减指数
        """
        self.prx = send_power
        self.d = distance
        self.f = f
        self.n = n

    def calculation(self):
        # 返回接受功率
        ptx = (self.prx * ((4*math.pi*self.f)**2)*math.pow(self.d, self.n)) / C**2

        return ptx


class Interference:
    def __init__(self, p, g, d, f, n):
        """
        干扰功率模型
        :param p: 干扰机有效发送功率
        :param g: 天线增益系数
        :param d: 干扰机到接收机的距离
        :param f: 与接收机相重叠的频率中频
        :param n: 衰减指数
        """
        self.p = p
        self.g = g
        self.d = d
        self.f = f
        self.n = n

    def calculation(self):
        pi = self.p * self.g * (C**2 / (4 * math.pi * self.f)**2 * math.pow(self.d, self.n))

        return pi


class SNR:
    def __init__(self):
        pass


def create_data(n, m):
    # 随机生成生成发送机的发送功率
    p = torch.randint(SEND_POWER_MIN, SEND_POWER_MAX, (int(n/2), ))
    p_zero = torch.zeros(int(n/2), )
    p = torch.cat([p, p_zero])
    p = p.detach().numpy().tolist()
    random.shuffle(p)

    # 随机生成干扰机发射功率
    jammer1 = torch.ones(int(m/2), ) * 800
    jammer2 = torch.ones(int(m/2), ) * 1000
    jammer = torch.cat([jammer1, jammer2])
    jammer = jammer.detach().numpy().tolist()
    random.shuffle(jammer)
    return p, jammer


def test():
    point = Simulation(1000, 'circle', 1000, 1000, 100)
    point.plot_inter_and_outer_point()
    p, jammer = create_data(len(point.inter_position), len(point.outer_position))
    re = Receiver(p, point.inter_position, 40000000, 4)


if __name__ == '__main__':
    test()
