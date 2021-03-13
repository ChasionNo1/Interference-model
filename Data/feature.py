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


# 特征的序列化
def write_files(data, path):
    with open(path, 'wb') as f:
        plk.dump(data, f)


def read_file(path):
    with open(path, 'rb') as f2:
        info = plk.load(f2)
        print(info)


def plot(x, y):
    plt.scatter(x, y)
    plt.axis('equal')
    plt.show()


def judge_equals():
    pass


def create_hyperedges(inx_list, dis_map, features):
    """
    超边构造：有顶点的k邻域，判断是否满足相互干扰的条件
           1、遍历整个inx_list, [ 0 15 17 13 12]
           2、分别计算每个顶点收到来自邻域的干扰总和，判断是否大于阈值
           3、外部干扰

           4、信道噪声干扰
     """

    hyperedges = []
    dis_map = dis_map[:, 1:]
    # print(dis_map)
    for i in range(len(inx_list)):
        interference = []
        point = inx_list[i]
        for j in range(len(point)):
            # 计算每个顶点受到其他顶点的干扰和，如果每个干扰和大于阈值，则建立超边
            temp = np.delete(point, j)
            feat_temp = [features[x][6] for x in temp]
            dis_temp = dis_map[j]
            result = sum([i / j for i, j in zip(feat_temp, dis_temp)])
            interference.append(result)
        # print(interference)
        a = np.array(interference)
        b = np.where(a > 5)[0]
        if len(b) == 4:
            hyperedges.append(point)
    print(hyperedges)


def create_feature():
    point = Simulation(5, 'circle', 20, 16, 5)
    point.plot_inter_and_outer_point()
    data = []
    for i in range(len(point.inter_position)):
        # id 属性 工作状态 等级 信道 坐标 干扰能力
        temp = [i + 1, 0, np.random.randint(0, 3), np.random.randint(0, 9), np.random.randint(0, 9), point.inter_position[i], np.random.randint(1, 7)]
        data.append(temp.copy())

    for j in range(len(point.outer_position)):
        temp2 = [len(point.inter_position) + j + 1, 1, point.outer_position[j], np.random.randint(0, 5)]
        data.append(temp2)
    # print(data)
    point_x = point.inter_position[:, 0]
    point_y = point.inter_position[:, 1]
    dis_map = []
    # 索引字典
    inx_dict = {j: i for i, j in enumerate(zip(point_x, point_y))}
    # print(inx_dict)
    # print(point.inter_position)
    for i in range(len(point.inter_position)):
        temp3 = euclidean_distances(point.inter_position, [point.inter_position[i]])
        dis_map.append(temp3.copy())

    # print(dis_map)
    # 选取与顶点u最接近的k个顶点，组成一个超边
    k_dis, inx = torch.topk(torch.Tensor(dis_map), 5, dim=1, largest=False)
    inx_list = inx.numpy().reshape(inx.shape[0], -1)
    k_dis = k_dis.numpy().reshape(k_dis.shape[0], -1)
    # 这里的ind_list其实就是k邻接的结果
    # print(inx_list)
    create_hyperedges(inx_list, k_dis, data)



    # plot(point_x, point_y)


if __name__ == '__main__':
    create_feature()




