# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/14 22:04
# @Author    :   Chasion
# Description: 加载数据集
import pickle as plk


def load_data():
    print('loading_datasets')
    names = ['edge_list', 'feats', 'label']
    objects = []
    for i in range(len(names)):
        with open(r'D:\graph_code\Interference-model\communication_model\Data\{}.content'.format(names[i]), 'rb')as f:
            objects.append(plk.load(f, encoding='latin1'))

    edge_list, feats, label = tuple(objects)
    feats = feats.astype('float32')
    print(feats[0])
    N = feats.shape[0]
    idx_train = [i for i in range(int(N * 0.5))]
    idx_val = [i for i in range(int(N * 0.5), int(N * 0.7))]
    idx_test = [i for i in range(int(N * 0.7), N)]

    return feats, label, idx_train, idx_val, idx_test, edge_list


if __name__ == '__main__':
    load_data()
