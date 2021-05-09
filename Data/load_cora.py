# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/25 9:58
# @Author    :   Chasion
# Description:
"""
created by weiyx15 @ 2019.1.4
Cora dataset interface
"""

import random
import numpy as np
from config import get_config
# from utils.construct_hypergraph import edge_to_hyperedge
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    """
    Copied from gcn
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def load_citation_data(cfg):
    """
    Copied from gcn
    citeseer/cora/pubmed with gcn split

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    '''
        Loads input data from gcn/data directory
        ind.dataset_str.x => 训练实例的特征向量，是scipy.sparse.csr.csr_matrix类对象，shape:(140, 1433)
        ind.dataset_str.tx => 测试实例的特征向量,shape:(1000, 1433)
        ind.dataset_str.allx => 有标签的+无无标签训练实例的特征向量，是ind.dataset_str.x的超集，shape:(1708, 1433)

        ind.dataset_str.y => 训练实例的标签，独热编码，numpy.ndarray类的实例，是numpy.ndarray对象，shape：(140, 7)
        ind.dataset_str.ty => 测试实例的标签，独热编码，numpy.ndarray类的实例,shape:(1000, 7)
        ind.dataset_str.ally => 对应于ind.dataset_str.allx的标签，独热编码,shape:(1708, 7)

        ind.dataset_str.graph => 图数据，collections.defaultdict类的实例，格式为 {index：[index_of_neighbor_nodes]}
        ind.dataset_str.test.index => 测试实例的id，2157行

    '''
    # 文件名
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    # 加载文件，添加到列表里
    for i in range(len(names)):
        with open(r"{}\ind.{}.{}".format(cfg['citation_root'], cfg['activate_dataset'], names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    # 将各个数据列表转为元组
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # 测试索引
    test_idx_reorder = parse_index_file(r"{}\ind.{}.test.index".format(cfg['citation_root'], cfg['activate_dataset']))
    # 测试索引排序，测试数据集的范围
    test_idx_range = np.sort(test_idx_reorder)
    # 如果加载的是citeseer数据集，需要补全孤立节点
    if cfg['activate_dataset'] == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # 这部分需要知道这些数据是如何供给给入口的
    # allx所有有标签和无标签的特征，tx是测试的特征、
    # vstack:按行拼接，列数必须相同，tolil:将稀疏矩阵转换为lil格式，提高查找速度
    features = sp.vstack((allx, tx)).tolil()
    # 对索引进行排序，按照排序后取特征
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # 对特征进行归一化
    features = preprocess_features(features)
    # todense返回矩阵
    features = features.todense()
    # 构建图结构
    G = nx.from_dict_of_lists(graph)
    # 图的邻接矩阵列表
    edge_list = G.adjacency_list()
    # 度
    degree = [0] * len(edge_list)
    # 添加自循环
    if cfg['add_self_loop']:
        for i in range(len(edge_list)):
            edge_list[i].append(i)
            degree[i] = len(edge_list[i])
    # 图的最大度
    max_deg = max(degree)
    # 图的平均度
    mean_deg = sum(degree) / len(degree)
    print(f'max degree: {max_deg}, mean degree:{mean_deg}')
    # 标签，和特征矩阵一样堆叠
    labels = np.vstack((ally, ty))
    # 排序？和特征矩阵一样
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # one-hot labels
    # 采样数，多少个顶点的标签
    n_sample = labels.shape[0]
    # 标签的类别数
    n_category = labels.shape[1]
    # 初始化一个np矩阵
    lbls = np.zeros((n_sample,))
    if cfg['activate_dataset'] == 'citeseer':
        n_category += 1  # one-hot labels all zero: new category
        for i in range(n_sample):
            try:
                lbls[i] = np.where(labels[i] == 1)[0]  # numerical labels
            except ValueError:  # labels[i] all zeros
                lbls[i] = n_category + 1  # new category
    else:
        for i in range(n_sample):
            lbls[i] = np.where(labels[i] == 1)[0]  # numerical labels
    # 这部分全是列表
    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))
    print(edge_list)

    return features, lbls, idx_train, idx_val, idx_test, n_category, edge_list, edge_list


if __name__ == '__main__':
    path = r'D:\graph_code\Interference-model\Data\\cora'
    cfg = {'activate_dataset': 'cora',
           'citation_root': path,
           'add_self_loop': True,
           }
    load_citation_data(cfg)
