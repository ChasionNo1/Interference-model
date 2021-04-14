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
    adj = [adj[i] for i in range(len(adj))]
    feats = feats.astype('float32')
    labels = labels.astype('int32')
    labels = labels.tolist()
    # print(edge_dict)
    # print(labels)
    # print(feats[2])
    # print(labels)
    '''
    如何解决稀疏问题？不需要稀疏处理
    如何确定数据集的划分问题？
    '''
    # feats = sp.csc_matrix(feats, dtype=np.float32)
    idx_train = [i for i in range(200)]
    idx_val = [i for i in range(200, 300)]
    idx_test = [i for i in range(300, 500)]
    # print(idx_train)
    # print(labels[:9])
    # adj = np.array(adj)
    return feats, adj, labels, idx_train, idx_val, idx_test, edge_dict


if __name__ == '__main__':
    load_data()
