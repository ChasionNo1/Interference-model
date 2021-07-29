# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/6/21 21:18
# @Author    :   Chasion
# Description:
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import torch

np.random.seed(20)
x = np.random.random(size=(10, 2))
# print(x)
# kmeans的聚类数据说明，x_shape: (样本数，样本特征维数），10个样本进行聚类，每个样本的特征维度为1
kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
center = kmeans.cluster_centers_
# print(center[0])
# print(kmeans.labels_)
# plt.scatter(x[:, 0], x[:, 1])
# plt.scatter(center[:, 0], center[:, 1])
# plt.show()
dis = euclidean_distances(x, center)
# dis里计算的是每个点与两个中心点的距离
print(dis)
# 取这两个距离里最小的，
_, cluster_center_dict = torch.topk(torch.Tensor(dis), k=1, largest=False)
cluster_center_dict = cluster_center_dict.numpy()
# 距离聚类中心点最近的一个点索引
# [[1]
#  [1]
#  [1]
#  [1]
#  [0]
#  [1]
#  [1]
#  [0]
#  [1]
#  [0]]
print(cluster_center_dict)
point_label = kmeans.labels_
# print(point_label)
# 每个聚类别里有哪些点
point_in_which_cluster = [np.where(point_label == i)[0] for i in range(2)]
# [array([4, 7, 9], dtype=int64), array([0, 1, 2, 3, 5, 6, 8], dtype=int64)]
print(point_in_which_cluster)
# 通过采样，得到_N, n_center, kc)
"""
有现成的计算模块，edge_dict: [[0, 221, 346], [1, 186], [2, 380], [3], [4, 30], [5, 356, 444], [6, 173, 269],.....
只要将edge_dict构造出来即可
"""
edge_dict = []
for i in range(cluster_center_dict.shape[0]):
    temp = point_in_which_cluster[cluster_center_dict[i][0]].tolist()
    idx = temp.index(i)
    temp[idx], temp[0] = temp[0], temp[idx]
    edge_dict.append(temp.copy())

print(edge_dict)

