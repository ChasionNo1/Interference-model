from Data.simulation import Simulation
import numpy as np
import pickle as plk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import torch
import os
from utils.layer_utils import sample_ids
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


def create_hyperedges(feats):
    """
    # version 1.0
    超边构造：有顶点的k邻域，判断是否满足相互干扰的条件
           1、遍历整个inx_list, [ 0 15 17 13 12]
           2、分别计算每个顶点收到来自邻域的干扰总和，判断是否大于阈值
           3、外部干扰

           4、信道噪声干扰
     """
    # hyperedges = []
    # dis_map = dis_map[:, 1:]
    # # print(dis_map)
    # for i in range(len(inx_list)):
    #     interference = []
    #     point = inx_list[i]
    #     for j in range(len(point)):
    #         # 计算每个顶点受到其他顶点的干扰和，如果每个干扰和大于阈值，则建立超边
    #         temp = np.delete(point, j)
    #         feat_temp = [features[x][3] for x in temp]
    #         dis_temp = dis_map[j]
    #         result = sum([i / j for i, j in zip(feat_temp, dis_temp)])
    #         interference.append(result)
    #     # print(interference)
    #     a = np.array(interference)
    #     b = np.where(a > 5)[0]
    #     if len(b) == 4:
    #         hyperedges.append(point)
    # # print(hyperedges)

    """
    version 2.0
    在1.0版本中，超边大小是固定，而在2.0中是根据情况可变的。
    根据特征矩阵，每个顶点收到其他顶点的干扰特征值为1，则与顶点构成一个超边
    采用补齐方式，先找到最大的超边大小，然后采用补0的方式
    在顶点最后加入一个特征全为0的顶点
    """

    """
    version 3.0
    不再使用特征全0的顶点填充，而是从现有的顶点中选择重复出现
    一个是超边集中顶点个数的统一
    一个是顶点超边集个数的统一
    """
    hyperedges = []
    # N个顶点
    N = feats.shape[0]
    # 创建一个空的point
    for i in range(N):
        temp = []
        index = np.where(feats[i] > 0)[0]
        temp.append(i)
        temp = temp + index.tolist()
        hyperedges.append(temp.copy())
    # 对不满足超边大小的超边进行填充
    edge_dict = hyperedges
    print(edge_dict)
    hyperedges = [sample_ids(hyperedges[i], 10) for i in range(N)]
    print(hyperedges)
    return hyperedges, edge_dict


def cal_interference(e_data, dis):
    # 计算两两顶点之间的干扰值
    # 参数:干扰能力，两点的欧式距离
    # 干扰值列表
    interference_map = []
    # 获取顶点干扰能力值
    point_inter_value = [e_data[x][3] for x in range(len(e_data))]
    # 获取顶点个数
    for i in range(len(e_data)):
        m = point_inter_value[i]
        distance = np.delete(dis[i], i)
        temp = [m / n for n in distance]
        temp.insert(i, 0)
        # 判断阈值，改为二进制形式
        temp = np.array(temp)
        index = np.where(temp > 1)[0]
        one_hot = np.zeros(len(e_data))
        one_hot[index] = 1
        interference_map.append(one_hot.copy())

    # print(interference_map[0])
    return np.array(interference_map)


def cal_outer_jammers(point, jammers):
    # 计算外部干扰机的干扰值
    # 外部干扰机的干扰值， 外部干扰机与各个顶点的距离
    dis = []
    jammers_interference = []
    # 计算干扰机到各个顶点的距离
    for i in range(len(point.outer_position)):
        temp = euclidean_distances(point.inter_position, [point.outer_position[i]])
        dis.append(temp.copy())
    dis = np.array(dis).reshape(len(dis), -1)
    # 计算干扰值
    point_jammers_value = [jammers[x][3] for x in range(len(jammers))]
    for i in range(len(jammers)):
        m = point_jammers_value[i]
        temp = [m / n for n in dis[i]]
        # 这里同样转换为one-hot
        temp = np.array(temp)
        index = np.where(temp > 1)[0]
        one_hot = np.zeros(dis.shape[1])
        one_hot[index] = 1
        jammers_interference.append(one_hot.copy())
    jammers_interference = np.array(jammers_interference).T
    # print(jammers_interference)
    return jammers_interference


def noise():
    # 计算信道噪声干扰
    pass


def get_random_num(a, num, seed):
    np.random.seed(seed)
    return [np.random.randint(a) for _ in range(num)]


def create_adj(hyperedges):
    # 为每个顶点构造一个超边邻域：
    # 将包含这个顶点的所有超边放在一个超边集里
    # 如果顶点没有超边，如何处理？
    # 用字典来存放顶点超边集，键是顶点编号，值是超边集
    adj = {}
    """
    [[ 1  3 14  6 11]
     [ 2  9 10 17 15]
     [ 3  1 14  6 11]
     [ 6 11 14  1  5]
     [ 9  2 10 17  4]
     [11  6 14  5  1]
     [14  6 11  1  3]]*
    """
    # 先对超边集进行去重和编码
    # 可以先排序，再去重得
    # print(hyperedges)
    # print(hyperedges)
    edge_size = 0
    for i in range(len(hyperedges)):
        adj[i] = []
        for j in range(len(hyperedges)):
            if i in hyperedges[j]:
                adj[i].append(hyperedges[j].copy())
    # 需要对顶点的超边集大小统一
    for i in range(len(adj)):
        l = len(adj[i])
        r = get_random_num(l, 6-l, i)
        temp = [adj[i][k] for k in r]
        adj[i] += temp
    # for u in adj:
    #     if edge_size < len(adj[u]):
    #         edge_size = len(adj[u])
    # supp = [len(adj)] * len(adj[0][0])
    # for u in adj:
    #     adj[u] = adj[u] + [supp] * (edge_size - len(adj[u]))
    print(adj)
    return hyperedges, adj


def create_feature():
    point = Simulation(5, 'circle', 20, 16, 5)
    point.plot_inter_and_outer_point()
    # 用来存放环境信息
    environment_data = []
    jammers_data = []
    for i in range(len(point.inter_position)):
        # id 属性 坐标 干扰能力
        temp = [i, 0, point.inter_position[i], np.random.randint(1, 7)]
        environment_data.append(temp.copy())

    for j in range(len(point.outer_position)):
        temp2 = [len(point.inter_position) + j, 1, point.outer_position[j], np.random.randint(5, 12)]
        jammers_data.append(temp2)
    # print(environment_data)

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

    # plot(point_x, point_y)
    inner = cal_interference(environment_data, dis_map)
    outer = cal_outer_jammers(point, jammers_data)
    # 将内部干扰和干扰机对每个顶点的干扰结果拼接在一起
    feats = np.c_[inner, outer]
    # 超边构造，version 2.0
    hyperedges, edge_dict = create_hyperedges(inner)
    hyperedges, adj = create_adj(hyperedges)
    # 写入文件中
    write_files(feats, 'datasets/feats.content')
    # 标签
    sum_1 = inner.sum(axis=1)
    index1 = np.where(sum_1 > 1)
    one_hot = np.zeros(inner.shape[0])
    one_hot[index1] = 1
    sum_2 = outer.sum(axis=1)
    index2 = np.where(sum_2 > 1)
    one_hot[index2] = 1
    write_files(one_hot, 'datasets/labels.content')
    write_files(adj, 'datasets/adj.content')
    write_files(edge_dict, 'datasets/edge_dict.content')


if __name__ == '__main__':
    create_feature()




