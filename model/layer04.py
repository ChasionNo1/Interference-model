# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/10 16:31
# @Author    :   Chasion
# Description:   由特征构造超边

import torch
from torch import nn
import numpy as np
from communication_model.load_data import load_data
from utils.layer_utils import cos_dis, sample_ids_v2, sample_ids
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class Transform(nn.Module):
    def __init__(self, dim_in, k):
        super(Transform, self).__init__()
        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        # 激活函数，用来分类，一维卷积对特征进行排列组合
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        # 重写forward函数
        # 区域特征：N,k,d，顶点集和，N是batch大小， k是一个超边里有多少顶点，d是顶点特征维度
        N, k, _ = region_feats.size()
        # 对特征进行一维卷积 (N, k*k, 1)
        convd = self.convKK(region_feats)
        # reshape
        multiplier = convd.view(N, k, k)
        # 过激活函数
        multiplier = self.activation(multiplier)
        # 变换矩阵与特征做乘法
        transformed_feats = torch.matmul(multiplier, region_feats)
        return transformed_feats


class VertexConv(nn.Module):
    """
    顶点卷积层
    """
    def __init__(self, dim_in, k):
        super().__init__()
        # (N, k, d) -> (N, k, d)
        self.trans = Transform(dim_in, k)
        # (N, k, d) -> (N, 1, d)  经过一维卷积将k个顶点的特征转换为一个超边的特征
        self.convK1 = nn.Conv1d(k, 1, 1)

    def forward(self, region_feats):
        # 调用转换模块，将输入特征进行转换，得到转换特征
        transformed_feats = self.trans(region_feats)
        # 进行一维卷积
        convd = self.convK1(transformed_feats)
        # 将特征平铺，(N, 1, d)--> (N,d)
        pooled_feats = convd.squeeze(1)
        return pooled_feats


class EdgeConv(nn.Module):
    """
    超边卷积模块:
    由顶点卷积得到超边特征，超边特征集和，先经过MLP(attention)，得到一个权重，
    然后再将各个超边特征与权重dot，再加权平均得到质心顶点的特征
    """
    def __init__(self, dim_ft, hidden):
        super().__init__()
        # 多层感知机,linear也就是全连接层
        # Linear(in_features, out_features, bias=True)
        # 最终输出是1，那么这里就是对一个超边进行MLP
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, feats):
        """
        使用自我关注系数计算dim = -2时的加权平均值
        :param feats: (N, t, d)  t是超边个数，d是超边的特征维数
        :return: (N, d)  顶点特征
        """
        scores = []
        n_edges = feats.size(1)
        for i in range(n_edges):
            scores.append(self.fc(feats[:, i]))
        scores = torch.softmax(torch.stack(scores, 1), 1)
        return (scores * feats).sum(1)


class GraphConvolution(nn.Module):
    """
    gcn layer
    这是一个仿照图卷积的超图卷积层
    简单的图卷积层，先是对顶点特征进行线性，再过激活函数，再dropout，完成特征提取
    然后再进行特征聚合
    """
    def __init__(self, **kwargs):
        # **的作用是将传入的字典进行unpack，然后将字典中的值作为关键词参数传入函数中。
        # 使用 ** kwargs定义参数时，kwargs将会接收一个positional argument后所有关键词参数的字典。
        super(GraphConvolution, self).__init__()
        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']

    def _region_aggregate(self, feats, edge_dict):
        # print(feats.size())
        # print(len(edge_dict))
        # N,d ---- >  N,d
        # 这是对超图中的超边完成初步的特征提取，按照dim=0维度求平均，并堆叠起来
        N = len(edge_dict)
        # edge_dict是顶点的超边集，里面的元素是顶点，将这个超边集里的顶点特征求平均，按列求平均，仍然是d维，只是将多个顶点的特征平均为一个
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])

        return pooled_feats

    def forward(self, feats, edge_dict):
        # N,d
        # 特征是顶点特征，N是顶点个数，d是特征维数
        x = feats
        # 经过全连接层后顶点特征维数发生了变化，N,dim_out
        x = self.dropout(self.activation(self.fc(x)))
        # 特征聚合
        x = self._region_aggregate(x, edge_dict)
        return x


class Attention(nn.Module):
    def __init__(self, **kwargs):
        super(Attention, self).__init__()
        self.dim_in = kwargs['dim_in']
        self.w_q = nn.Linear(self.dim_in, 1)
        self.w_k = nn.Linear(self.dim_in, 1)
        self.w_v = nn.Linear(self.dim_in, 1)
        self.softmax = nn.Softmax(0)
        self.tanh = nn.Tanh()

    def forward(self, feats, edge_dict):
        """

        :param feats:
        :param edge_dict: [[0, 221, 346], [1, 186], [2, 380], [3], [4, 30], [5, 356, 444], [6, 173, 269],.....
        :return:
        """
        n_edge = len(edge_dict)
        edge_feats = []
        for i in range(n_edge):
            # 得到每个主属性的超边中的顶点个数
            n_point = len(edge_dict[i])
            if n_point == 1:
                edge_feats.append(feats[i].detach().numpy().tolist())
            else:
                edge_array = np.array(edge_dict[i])
                new = np.delete(edge_array, 0)
                ei = self.w_q(feats[edge_dict[i][0]]).unsqueeze(dim=0)
                ej = self.w_k(feats[new])
                eij = torch.multiply(ei, ej)
                aij = self.softmax(eij)
                feats[edge_dict[i][0]].add_(torch.mul(aij, feats[new]).sum(0).squeeze(dim=0))
                edge_feats.append(feats[edge_dict[i][0]].detach().numpy().tolist())
        return torch.LongTensor(edge_feats)


class DHGLayer(nn.Module):
    """
    这部分我要做什么？
    edge_list中代表着与质心顶点相连的其他顶点组成的超边，这个是有大有小，可以使用注意力机制采样填充
    knn可以使用
    kmeans我觉得可以使用距离来进行聚类，邻近节点的干扰情况会影响质心顶点
    """

    def __init__(self, **kwargs):
        super(DHGLayer, self).__init__()
        self.dim_in = kwargs['dim_in']
        # edge_list中采样个数
        self.ks = kwargs['structured_neighbor']
        # 最邻近构造，这里我觉得可以用距离的最邻近来构造
        self.kn = kwargs['nearest_neighbor']
        # kmeans作为超边的个数，
        # 特征值邻近代表着什么？如果两个特征非常接近，说明受干扰情况很接近，则在一个超边里是很可能的。
        self.kc = kwargs['cluster_neighbor']
        self.n_cluster = kwargs['n_cluster']
        self.n_center = kwargs['n_center']

        # warm_up 预热部分
        self.wu_knn = kwargs['wu_knn']
        self.wu_kmeans = kwargs['wu_kmeans']
        self.wu_struct = kwargs['wu_struct']

        self.vc_n = VertexConv(self.dim_in, self.kn)
        self.vc_c = VertexConv(self.dim_in, self.kc)
        self.vc_s = VertexConv(self.dim_in, self.ks)
        self.ec = EdgeConv(self.dim_in, hidden=self.dim_in//4)
        self.kmeans = None
        self.structure = None

        self.activation = kwargs['activation']
        self.dropout = nn.Dropout(p=0.5)
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=True)

    def structure_select(self, ids, feats, edge_dict):
        # if self.structure is None:
        #     t = Attention(dim_in=feats.size(1))
        #     edge_feats = t.forward(feats, edge_dict)
        #     self.structure = edge_feats
        # else:
        #     edge_feats = self.structure
        #
        # region_feats = edge_feats[ids]
        if self.structure is None:
            '''
            特征由来的过程：首先是超边中k个顶点，采样组成一个k*d维的特征矩阵，然后再经过顶点卷积，将这k个顶点的特征融合维一个超边的特征
            然后再将顶点u邻域adj(u)超边聚合成u的一个特征xu，最后经过线性和非线性变换。
            '''
            _N = feats.size(0)
            # ks：number of sampled nodes in graph adjacency
            '''
            edge_dict=
            _N:是adj(u)中超边的个数
            '''
            # idx是所有超边中每个顶点索引构成的列表(_N,ks)
            # 这里进行采样，也是填充，128
            idx = torch.LongTensor([sample_ids(edge_dict[i], self.ks) for i in range(_N)])  # (_N, ks)
            self.structure = idx
        else:
            idx = self.structure
            # print(edge_dict[0])
            # print(self.structure[0])
            #  torch.Size([2708, 128]) 2708个点 128采样点
            # print('structure', self.structure.size())

            # ids是在训练/有效/测试期间选择的索引，用来索引要用到的顶点
        idx = idx[ids]
        # idx:[_N,ks], N是超边个数，ks采样顶点数
        N = idx.size(0)
        d = feats.size(1)  # 特征维数
        # 采样后的组成的特征[N, ks, d]， N个超边中ks个顶点的d维特征
        region_feats = feats[idx.view(-1)].view(N, self.ks, d)  # (N, ks, d)

        return region_feats

    def nearest_select(self, ids, feats):
        dis = cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        idx = idx[ids]
        N = len(idx)
        d = feats.size(1)
        nearest_feats = feats[idx.view(-1)].view(N, self.kn, d)

        return nearest_feats

    def cluster_select(self, ids, feats):
        """

        :param ids:
        :param feats:
        :return:
        """
        if self.kmeans is None:

            _N = feats.size(0)
            np_feats = feats.detach().numpy()
            kmeans = KMeans(n_clusters=self.n_cluster, random_state=0, n_jobs=-1).fit(np_feats)
            centers = kmeans.cluster_centers_
            dis = euclidean_distances(np_feats, centers)
            _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_cluster, largest=False)
            cluster_center_dict = cluster_center_dict.numpy()
            point_labels = kmeans.labels_
            # 顶点在哪一个聚类里
            point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]
            # 采样点的kc个临近聚类团体最为它的超边
            idx = torch.LongTensor([[sample_ids_v2(point_in_which_cluster[cluster_center_dict[point][i]], self.kc)
                                     for i in range(self.n_center)] for point in range(_N)])  # (_N, n_center, kc)
            self.kmeans = idx
        else:
            idx = self.kmeans

        idx = idx[ids]
        N = idx.size(0)
        d = feats.size(1)
        # 融合聚类特征
        cluster_feats = feats[idx.view(-1)].view(N, self.n_center, self.kc, d)

        return cluster_feats  # (N, n_center, kc, d)

    def _fc(self, x):
        return self.activation(self.fc(self.dropout(x)))

    def forward(self, ids, feats, edge_dict, epo):
        hyperedge = []
        if epo >= self.wu_kmeans:
            c_feats = self.cluster_select(ids, feats)
            for c_idx in range(c_feats.size(1)):
                xc = self.vc_c(c_feats[:, c_idx, :, :])
                xc = xc.view(len(ids), 1, feats.size(1))
                hyperedge.append(xc)

        if epo >= self.wu_knn:
            n_feats = self.nearest_select(ids, feats)
            xn = self.vc_n(n_feats)
            xn = xn.view(len(ids), 1, feats.size(1))
            hyperedge.append(xn)

        if epo >= self.wu_kmeans:
            s_feat = self.structure_select(ids, feats, edge_dict)
            # self.vc_s = VertexConv(self.dim_in, self.ks)    # structured trans
            xs = self.vc_s(s_feat)
            xs = xs.view(len(ids), 1, feats.size(1))  # (N, 1, d)
            hyperedge.append(xs)

        x = torch.cat(hyperedge, dim=1)
        x = self.ec(x)
        x = self._fc(x)
        return x


if __name__ == '__main__':
    feats, labels, idx_train, idx_val, idx_test, edge_dict = load_data()
    feats = torch.Tensor(feats)
    d = feats.size()[1]
    # a = Attention(dim_in=n)
    # a.forward(feats, edge_dict)
    relu = nn.Sigmoid()
    p = {
        'dim_in': d,
        'dim_out': 2,
        'nearest_neighbor': 10,
        'cluster_neighbor': 5,
        'n_cluster': 30,
        'wu_knn': 0,
        'wu_kmeans': 10,
        'wu_struct': 5,
        'activation': relu,
    }
    layer = DHGLayer(dim_in=d, dim_out=2, structured_neighbor=10, nearest_neighbor=10, cluster_neighbor=10, n_cluster=30, n_center=5, wu_knn=0, wu_kmeans=10, wu_struct=5, activation=relu)
    layer.forward(ids=idx_train, feats=feats, edge_dict=edge_dict, epo=15)
