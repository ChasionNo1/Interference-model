# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/14 22:04
# @Author    :   Chasion
# Description: 加载数据集
import pickle as plk


def load_data():
    print('loading_datasets')
    # names = ['train_edge_list', 'train_feats', 'train_label']
    # 'edge_list', 'feats', 'label'
    # 'prediction_edge_list', 'prediction_feats', 'prediction_label'
    names = ['prediction_edge_list', 'prediction_feats', 'prediction_label']
    objects = []
    for i in range(len(names)):
        with open(r'D:\graph_code\Interference-model\communication_model\Data\25\{}.content'.format(names[i]), 'rb')as f:
            objects.append(plk.load(f, encoding='latin1'))

    edge_list, feats, label = tuple(objects)
    feats = feats.astype('float32')
    print(feats.shape)
    N = feats.shape[0]
    # 计算平均度和最大度
    print(edge_list)
    degree = []
    max = 0
    for i in range(len(edge_list)):
        l = len(edge_list[i])
        if l > max:
            max = l
        degree.append(l)
    print(sum(degree)/500)
    print(max)
    idx_train = [i for i in range(int(N * 0.8))]
    idx_val = [i for i in range(int(N * 0.8), int(N * 0.9))]
    idx_test = [i for i in range(int(N * 0.9), N)]

    return feats, label, idx_train, idx_val, idx_test, edge_list


if __name__ == '__main__':
    load_data()
