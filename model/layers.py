import torch
from torch import nn
import numpy as np
"""
网络层
"""


class Transform(nn.Module):
    """
    这部分是得到转换矩阵与顶点特征的乘积，返回转换特征，
    这个特征后续只需要再进行一维卷积和平铺就可以得到超边特征了
    """

    def __init__(self, dim_in, k):
        super().__init__()
        # 一维卷积：k是输入通道，k*k是输出通道
        self.convKK = nn.Conv1d(k, k*k, dim_in, groups=k)
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
        # N,d ---- >  N,d
        # 这是对超图中的超边完成初步的特征提取，按照dim=0维度求平均，并堆叠起来
        N = feats.size()[0]
        # edge_dict是顶点的超边集，里面的元素是顶点，将这个超边集里的顶点特征求平均，按列求平均，仍然是d维，只是将多个顶点的特征平均为一个
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])

        return pooled_feats

    def forward(self, ids, feats, edge_dict, G, ite):
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
    动态超图卷积层
    """
    def __init__(self, **kwargs):
        super(DHGLayer, self).__init__()
        # 超边集中顶点的采样个数
        self.ks = kwargs['structured_neighbor']
        # 聚类的个数
        self.n_cluster = kwargs['n_cluster']
        # 一个顶点的邻接超边的个数
        self.n_center = kwargs['n_center']
        # k邻近中取得顶点个数
        self.kn = kwargs['nearest_neighbor']
        # k均值聚类每个类别采样的顶点个数
        self.kc = kwargs['cluster_neighbor']
        # warm-up parameter 预热参数，如果没有预热，前几个训练步骤的性能将不稳定，因为在前几个步骤中，超图构造基于当前的特征图。









