# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/15 19:07
# @Author    :   Chasion
# Description:
import pickle as plk


def load_data2():
    print('loading_datasets')
    names = ['prediction_edge_list', 'prediction_feats', 'prediction_label']
    objects = []
    for i in range(len(names)):
        with open(r'D:\graph_code\Interference-model\communication_model\Data\{}.content'.format(names[i]), 'rb')as f:
            objects.append(plk.load(f, encoding='latin1'))

    edge_list, feats, label = tuple(objects)
    feats = feats.astype('float32')
    # print(feats[0])
    N = feats.shape[0]
    # idx_train = [i for i in range(int(N * 0.6))]
    # idx_val = [i for i in range(int(N * 0.6), int(N * 0.8))]
    idx_test = [i for i in range(0, N)]

    return feats, label, idx_test, edge_list