import torch
from torch import nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from utils.layer_utils import sample_ids, sample_ids_v2, cos_dis
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
        self.wu_knn = kwargs['wu_knn']
        self.wu_kmeans = kwargs['wu_kmeans']
        self.wu_struct = kwargs['wu_struct']
        self.vc_sn = VertexConv(self.dim_in, self.ks + self.kn)
        self.vc_s = VertexConv(self.dim_inm, self.ks)
        self.vc_n = VertexConv(self.dim_in, self.kn)
        self.vc_c = VertexConv(self.dim_in, self.kc)
        self.ec = EdgeConv(self.dim_in, hidden=self.dim_in/4)
        self.kmeans = None
        self.structure = None

    def _vertex_conv(self, func, x):
        return func(x)

    def _structure_select(self, ids, feats, edge_dict):
        """
        这部分是从超边集中采样ks个点，然后将这个些点组成一个特征矩阵
        :param ids: 在训练/有效/测试期间选择的索引，
        :param feats: 所有顶点组成的特征矩阵
        :param edge_dict:
        :return:邻域图
        """
        if self.structure is None:
            # feats: _N, d,  _N是顶点个数，
            _N = feats.size()[0]
            # 对所有顶点的超边集进行采样，采样ks个点
            idx = torch.LongTensor([sample_ids(edge_dict[i], self.ks) for i in range(_N)])
            self.structure = idx
        else:
            idx = self.structure
        # 采样过的邻域图，再进行，训练、测试等数据集的划分索引
        idx = idx[ids]
        # 得到划分的顶点个数
        N = idx.size()[0]
        # 特征维数
        d = feats.size(1)
        # reshape成一个N,ks,d的矩阵
        region_feats = feats[idx.view(-1)].view(N, self.ks, d)
        return region_feats

    """
    knn和kmeans
    这部分在超图构造的时候使用，但是我的数据已经构造完毕了
    """

    def _nearest_select(self, ids, feats):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: mapped nearest neighbors    最近邻域图
        """
        # 计算(N, d)N个超边的余弦距离
        dis = cos_dis(feats)
        # 沿着dim=1维度给出dis中kn个最大值，返回一个元组，values，indices
        _, idx = torch.topk(dis, self.kn, dim=1)
        # 得到最大的kn个元素索引，聚合这个kn个超边的特征，组成N，kn，d的特征矩阵
        idx = idx[ids]
        N = len(idx)
        d = feats.size(1)
        nearest_feature = feats[idx.view(-1)].view(N, self.kn, d)  # (N, kn, d)
        return nearest_feature

    def _cluster_select(self, ids, feats):
        """
        compute k-means centers and cluster labels of each node
        return top #n_cluster nearest cluster transformed features
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: top #n_cluster nearest cluster mapped features
        """
        if self.kmeans is None:
            _N = feats.size(0)
            # detach():阻止反向传播的，cpu():将数据复制到cpu中，将tensor转换为numpy数组
            np_feats = feats.detach().cpu().numpy()
            # 生成的聚类数，random_state：整形或 numpy.RandomState 类型，可选
            # 用于初始化质心的生成器（generator）。如果值为一个整数，则确定一个seed。此参数默认值为numpy的随机数生成器。
            # n_jobs：整形数。　指定计算所用的进程数。内部原理是同时进行n_init指定次数的计算。
            # （１）若值为 -1，则用所有的CPU进行运算。若值为1，则不进行并行运算，这样的话方便调试。
            kmeans = KMeans(n_clusters=self.n_cluster, random_state=0, n_jobs=-1).fit(np_feats)
            # kmeans的属性，聚类的中心坐标向量，[n_clusters, n_features] (聚类中心的坐标)
            centers = kmeans.cluster_centers_
            # 特征矩阵与聚类中心的欧式距离，
            dis = euclidean_distances(np_feats, centers)
            # 得到self.n_center个最大值
            _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)
            cluster_center_dict = cluster_center_dict.numpy()
            # 每个顶点的标签
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

    def _edge_conv(self, x):
        return self.ec(x)

    def _fc(self, x):
        return self.activation(self.fc(self.dropoutx))

    def forward(self, ids, feats, edge_dict, G, ite):
        """
        重写函数
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

        :param ids:在训练/有效/测试期间选择的索引
        :param feats:
        :param edge_dict:
        :param G:
        :param ite:
        :return:采样yu
        """
        hyperedges = []
        # 带有预热参数，
        if ite >= self.wu_kmeans:
            c_feat = self._cluster_select(ids, feats)
            for c_idx in range(c_feat.size(1)):
                xc = self._vertex_conv(self.vc_c, c_feat[:, c_idx, :, :])
                xc = xc.view(len(ids), 1, feats.size(1))
                hyperedges.append(xc)

        if ite >= self.wu_knn:
            n_feat = self._nearest_select(ids, feats)
            xn = self._vertex_conv(self.vc_n, n_feat)
            xn = xn.view(len(ids), 1, feats.size(1))
            hyperedges.append(xn)

        if ite >= self.wu_struct:
            s_feat = self._structure_select(ids, feats, edge_dict)
            xs = self._vertex_conv(self.vc_s, s_feat)
            xs = xs.view(len(ids), 1, feats.size(1))
            hyperedges.append(xs)

        # 顶点卷积完成，开始合并超边特征进行超边卷积
        x = torch.cat(hyperedges, dim=1)
        x = self._edge_conv(x)
        x = self._fc(x)
        return x


class HGNN_conv(nn.Module):
    """
    a HGNN layer
    """
    def __init__(self, **kwargs):
        super(HGNN_conv, self).__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has-bias'])
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']

    def forward(self, ids, feats, edge_dict, G, ite):
        x = feats
        x = self.activation(self.fc(x))
        x = G.matmul(x)
        x = self.dropout(x)
        return x


















