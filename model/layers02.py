"""
version 1.0
对原有代码进行精简，去掉不用的功能以及配置文件载入方式
"""
import torch
from torch import nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from Data.load_data import load_data


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
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
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

    def forward(self, ids, feats, edge_dict, adj):
        # N,d
        # 特征是顶点特征，N是顶点个数，d是特征维数
        x = feats
        # 经过全连接层后顶点特征维数发生了变化，N,dim_out
        x = self.dropout(self.activation(self.fc(x)))
        # 特征聚合
        x = self._region_aggregate(x, edge_dict)
        return x


class DHGLayer(GraphConvolution):
    """
    这部分功能的重写，去掉knn和kmeans部分

    先描述一下我的特征构造：
    1、feats:每个顶点收到其他顶点的干扰以及干扰机的干扰
    2、hyperedges:经过判别法则，构成的超边
    3、labels：标签
    4、adj：每个顶点的超边集


    框架的流程：
    输入顶点采样特征xu，超图结构G
    伪代码:
    建立一个空列表存放超边
    对edge_list进行遍历，对每个顶点的超边集进行：
        顶点采样
        顶点卷积
        添加顶点卷积结果到超边list中
    （顶点卷积，将超边集里的各个顶点特征融合为超边特征，一个顶点包含在多个超边中，会得到多个超边特征）
    然后将每个顶点的超边特征堆叠起来
    进行超边卷积
    过非线性，融合为顶点的新特征


    """
    def __init__(self, **kwargs):
        super(DHGLayer, self).__init__(**kwargs)
        # 这里不需要再构造超边，因此不需要采样点，聚类，knn
        # 加入预热参数
        # 没有采样，所以没有structure,也没有采用kmeans
        self.ec = EdgeConv(self.dim_in, hidden=self.dim_in//4)
        self.vc = EdgeConv(self.dim_in, 9)

    def _vertex_conv(self, func, x):
        return func(x)

    def _edge_conv(self, x):
        return self.ec(x)

    def _fc(self, x):
        return self.activation(self.fc(self.dropout(x)))

    def forward(self, ids, feats, edge_dict, adj):
        """

        :param ids: 训练或者验证、测试的索引列表
        :param feats: N,d 特征
        :param adj: N个顶点的超边集
        :return:
        """
        hyperedges_feats = []
        adj = adj[ids]
        N = adj.size(0)
        s = adj.size(1)
        k = adj.size(2)
        d = feats.size(1)
        reshape_feats = feats[adj.view(-1)].view(N, s, k, d)
        # print(reshape_feats.size())
        for i in range(reshape_feats.size(1)):
            conv_result = self._vertex_conv(self.vc, reshape_feats[:, i, :, :])
            conv_result = conv_result.view(len(ids), 1, feats.size(1))
            hyperedges_feats.append(conv_result)
        x = torch.cat(hyperedges_feats, dim=1)
        x = self._edge_conv(x)
        x = self._fc(x)
        # print('output', x)
        return x


if __name__ == '__main__':
    feats, adj, labels, idx_train, idx_val, idx_test, edge_dict = load_data()
    feats = torch.Tensor(feats)
    adj = torch.LongTensor(adj)
    a = GraphConvolution(dim_in=feats.size()[1], dim_out=feats.size()[1], has_bias=True, activation=nn.ReLU())
    # 图卷积作为输入层
    x = a.forward(idx_train, feats, edge_dict, adj)
    d = DHGLayer(dim_in=feats.size()[1], dim_out=2, has_bias=True, activation=nn.Sigmoid())
    d.forward(idx_train, x, edge_dict, adj)



















