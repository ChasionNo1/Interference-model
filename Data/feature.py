"""
特征构造

"""
from Data.simulation import Simulation
import numpy as np
import pickle as plk


# 获取各个顶点的坐标

def write_files(data, path):
    with open(path, 'wb') as f:
        plk.dump(data, f)


def read_file(path):
    with open(path, 'rb') as f2:
        info = plk.load(f2)
        print(info)


def create_feature():
    point = Simulation(5, 'circle', 20, 16, 5)
    point.plot_inter_and_outer_point()
    data = []
    for i in range(len(point.inter_position)):
        # id 属性 工作状态 等级 信道 坐标 干扰能力
        temp = [i + 1, 0, np.random.randint(0, 3), np.random.randint(0, 9), np.random.randint(0, 9), point.inter_position[i], np.random.randint(0, 5)]
        data.append(temp.copy())
    for j in range(len(point.outer_position)):
        temp2 = [len(point.inter_position) + j + 1, 1, point.outer_position[j], np.random.randint(0, 5)]
        data.append(temp2.copy())
    print(data)


create_feature()




