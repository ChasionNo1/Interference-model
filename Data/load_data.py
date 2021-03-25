import numpy as np
import pickle as plk
import torch
import scipy.sparse as sp



"""
这部分直接从数据集中加载数据 
"""


def load_data():
    print('loading datasets')
    names = ['adj', 'edge_dict', 'feats', 'labels']
    objects = []
    for i in range(len(names)):
        with open(r'D:\graph_code\Interference-model\Data\datasets\{}.content'.format(names[i]), 'rb') as f:
            objects.append(plk.load(f, encoding='latin1'))

    # print(objects)
    adj, edge_dict, feats, labels = tuple(objects)
    # print(adj)
    # print(feats)
    # print(labels)
    '''
    如何解决稀疏问题？不需要稀疏处理
    如何确定数据集的划分问题？
    '''
    # feats = sp.csc_matrix(feats, dtype=np.float32)
    # print(feats)
    idx_train = range(8)
    idx_val = range(9, 15)
    idx_test = range(15, 20)

    return feats, adj, labels, idx_train, idx_val, idx_test, edge_dict


if __name__ == '__main__':
    load_data()
