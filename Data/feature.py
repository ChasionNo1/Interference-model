from Data.simulation import Simulation
import numpy as np
import pickle as plk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import torch
"""
特征构造
"""


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
        data.append(temp2)
    # print(data)
    point_x = point.inter_position[:, 0]
    point_y = point.inter_position[:, 1]
    dataMap = []
    # 索引字典
    inx_dict = {j: i for i, j in enumerate(zip(point_x, point_y))}
    # print(inx_dict)
    # print(point.inter_position)
    for i in range(len(point.inter_position)):
        temp3 = euclidean_distances(point.inter_position, [point.inter_position[i]])
        dataMap.append(temp3.copy())

    # print(dataMap)
    # 选取与顶点u最接近的k个顶点，组成一个超边
    _, inx = torch.topk(torch.Tensor(dataMap), 5, dim=1, largest=False)
    inx_list = inx.numpy().reshape(inx.shape[0], -1)
    # 这里的ind_list其实就是初步的超边
    print(inx_list)
    # print(inx)
    # point_x_c = kmeans.cluster_centers_[:, 0]
    # point_y_c = kmeans.cluster_centers_[:, 1]
    # print(kmeans.cluster_centers_)
    # print(kmeans.labels_)
    plt.scatter(point_x, point_y)
    # plt.scatter(point_x_c, point_y_c)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    create_feature()




