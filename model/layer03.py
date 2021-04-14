"""
layer03:
这部分去掉采样填充的方式，使用多头注意力来解决超边大小不一的问题
去掉图卷积作为输入层
引入预热部分
data：2021/4/6
"""
import torch
from torch import nn
from Data.load_data import load_data
import numpy as np


class VertexConv(nn.Module):
    def __init__(self, dim_in):
        super(VertexConv, self).__init__()
        self.w_q = nn.Linear(dim_in, 1)
        self.w_k = nn.Linear(dim_in, 1)
        self.w_v = nn.Linear(dim_in, 1)
        self.softmax = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

    def forward(self, feats, edge_dict):
        """
        想在这部分替代顶点卷积
        将一个超边集内的顶点聚合为一个超边特征
        :param feats: input features
        :param edge_dict: the Main Attributes
        :return:
        """

        """
        edge_dict:
        [[0, 29, 31], [1, 33, 50, 56, 58, 64], [2, 0, 10, 13, 21, 25, 29, 31, 40, 65], [3, 27, 46, 56], [4, 45, 47, 59, 76], [5, 19, 32, 51, 78], [6], [7, 20, 37, 68], [8, 15, 24], [9, 17, 22, 23, 54, 55, 71],
        """
        # 得到edge_dict的情况，edge_dict是一个列表，因为维度不同，无法转为numpy或者tensor
        n_edge = len(edge_dict)
        score = []
        edge_feats = []
        for i in range(n_edge):
            temp = []
            # 得到每个主属性的超边中的顶点个数
            n_point = len(edge_dict[i])
            if n_point == 1:
                edge_feats.append(feats[i].detach().numpy().tolist())
            else:
                hyperedge_feat = torch.zeros(feats.size(1))
                for j in range(n_point):
                    edge_array = np.array(edge_dict[i])
                    new = np.delete(edge_array, j)
                    ei = self.w_q(feats[edge_dict[i][j]]).unsqueeze(dim=0)
                    ej = self.w_k(feats[new])
                    eij = torch.multiply(ei, ej)
                    aij = self.softmax(eij)
                    di = self.tanh(torch.multiply(aij, self.w_v(feats[new])).sum(0))
                    hyperedge_feat.add_(torch.multiply(di, feats[edge_dict[i][j]]))

                edge_feats.append(hyperedge_feat.detach().numpy().tolist())

        return torch.Tensor(edge_feats)


class EdgeConv(nn.Module):
    def __init__(self, dim_in, hidden):
        super(EdgeConv, self).__init__()
        self.fc = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, feats):
        re_feats = []
        for i in range(len(feats)):
            temp = []
            ft = torch.Tensor(feats[i]).unsqueeze(dim=0)
            for j in range(len(feats[i])):
                temp.append([self.fc(ft)[0][j].item()])
            temp = torch.softmax(torch.Tensor(temp), 0)

            h = (temp * ft).sum(1)
            re_feats.append(h)
        re_feats = torch.cat(re_feats, dim=0)

        return re_feats


class GraphConv(nn.Module):
    def __init__(self, dim_in):
        super(GraphConv, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(dim_in, dim_in, bias=True)
        self.activation = nn.ReLU()

    def region_aggreagte(self, feats, edge_dict):
        N = feats.size(0)
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])

        return pooled_feats

    def forward(self, idx, feats, edge_dict, sadj, epoch):
        x = feats
        x = self.dropout(self.activation(self.fc(x)))
        x = self.region_aggreagte(x, edge_dict)
        return x


class DHGLayerV1(nn.Module):
    def __init__(self, dim_in):
        super(DHGLayerV1, self).__init__()
        self.vc = VertexConv(dim_in)
        self.ec = EdgeConv(dim_in, hidden=dim_in//4)
        # 预热部分，主属性的聚合
        self.wu_e = 0
        # 超边集属性的聚合
        self.wu_a = 5
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(dim_in, 2, bias=True)

    def _fc(self, x):
        return self.activation(self.fc(self.dropout(x)))

    def forward(self, ids, feats, edge_dict, adj, epoch):
        """

        :param ids: 训练、验证、测试的数据集划分
        :param feat:
        :param edge_dict:
        :param adj: [[[0, 29, 31], [2, 0, 10, 13, 21, 25, 29, 31, 40, 65], [13, 0, 2, 10, 21, 25, 29, 31, 40, 65, 70], [21, 0, 2, 13, 29, 31, 40], [29, 0, 2, 13, 21, 31, 65], [31, 0, 2, 13, 21, 29, 40, 65]],
        :param epoch: 训练回合，加入预热部分
        :return:
        """
        hyperedge_feats = []
        tempa = []
        tempb = []
        edge_dict = edge_dict[ids[0]: ids[-1]+1]
        adj = adj[ids[0]: ids[-1]+1]
        d = feats.size(1)
        if epoch >= self.wu_e:
            xn = self.vc(feats, edge_dict)
            xn = xn.view(len(ids), 1, d)
            tempa = xn.detach().numpy().tolist()

        if epoch >= self.wu_a:

            for i in range(len(adj)):
                xe = self.vc(feats, adj[i])
                # torch.Size([6, 91]) 这个要和主属性的那个聚合在一起 --> [7, 91]
                tempb.append(xe.detach().numpy().tolist())

        if len(tempb) > 0:
            for i in range(len(tempa)):
                hyperedge_feats.append(tempa[i] + tempb[i])

        else:
            hyperedge_feats.append(tempa)

        # 合并后，维度不同，不能转换为张量，怎么办？

        x = self.ec(hyperedge_feats)
        x = self._fc(x)
        # print(x)

        return x


if __name__ == '__main__':
    feats, adj, labels, idx_train, idx_val, idx_test, edge_dict = load_data()
    n = feats.shape[0]
    d = feats.shape[1]
    feats = torch.Tensor(feats)
    layer = DHGLayerV1(dim_in=d)
    layer.forward(idx_train, feats, edge_dict, adj, 6)











