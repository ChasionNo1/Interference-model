# from model.layers02 import *
# from model.layer03 import *
from model.layer04 import *


# class DHGNN_v1(nn.Module):
#     """
#     动态超图神经网络：采用图卷积作为输入层
#     """
#     def __init__(self, **kwargs):
#         super(DHGNN_v1, self).__init__()
#
#         self.dim_feat = kwargs['dim_feat']
#         self.n_categories = kwargs['n_categories']
#         self.n_layers = kwargs['n_layers']
#         self.layer_spec = kwargs['layer_spec']
#         self.dims_in = [self.dim_feat] + self.layer_spec
#         self.dims_out = self.layer_spec + [self.n_categories]
#         self.activations = nn.ModuleList([nn.ReLU() for _ in range(self.n_layers - 1)] + [nn.Sigmoid()])
#         self.gcs = nn.ModuleList([GraphConvolution(
#             dim_in=self.dims_in[0],
#             dim_out=self.dims_out[0],
#             dropout_rate=kwargs['dropout_rate'],
#             activation=self.activations[0],
#             has_bias=kwargs['has_bias'])]
#             + [DHGLayer(
#             dim_in=self.dims_in[i],
#             dim_out=self.dims_out[i],
#             dropout_rate=kwargs['dropout_rate'],
#             activation=self.activations[i],
#             has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])
#
#     def forward(self, **kwargs):
#         ids = kwargs['ids']
#         feats = kwargs['feats']
#         adj = kwargs['adj']
#         edge_dict = kwargs['edge_dict']
#
#         x = feats
#         for i in range(self.n_layers):
#             x = self.gcs[i](ids, x, edge_dict, adj)
#         return x
#
#
# class DHGNN_v2(nn.Module):
#     """
#     使用多头注意力层代替顶点卷积，可以处理超边大小不一的问题
#     """
#     def __init__(self, dim_in):
#         super(DHGNN_v2, self).__init__()
#         self.gcs = nn.ModuleList([GraphConv(dim_in=dim_in)] + [DHGLayerV1(dim_in=dim_in)])
#
#     def forward(self, ids, feats, edge_dict, adj, epoch):
#         x = feats
#         for i in range(2):
#             x = self.gcs[i](ids, x, edge_dict, adj, epoch)
#         return x


class DHGNN_v3(nn.Module):
    """
    version 3.0
    """
    def __init__(self, **kwargs):
        super(DHGNN_v3, self).__init__()

        self.dim_feats = kwargs['dim_feats']
        self.dims_out = [self.dim_feats, 2]
        self.nearest_neighbor = 5
        self.cluster_neighbor = 5
        self.structured_neighbor = 15
        self.n_cluster = 20
        self.n_center = 1
        self.wu_knn = 0
        self.wu_kmeans = 10000
        self.wu_struct = 5
        self.activation = nn.ModuleList([nn.ReLU()] + [nn.Sigmoid()])
        '''
        [GraphConvolution(
            dim_in=self.dim_feats,
            dim_out=self.dims_out[0],
            activation=self.activation[0]
        )]
        + 
        '''
        self.gcs = nn.ModuleList([DHGLayer(
            dim_in=self.dim_feats,
            dim_out=self.dims_out[1],
            nearest_neighbor=self.nearest_neighbor,
            cluster_neighbor=self.cluster_neighbor,
            structured_neighbor=self.structured_neighbor,
            n_cluster=self.n_cluster,
            n_center=self.n_center,
            wu_knn=self.wu_knn,
            wu_struct=self.wu_struct,
            wu_kmeans=self.wu_kmeans,
            activation=self.activation[1]
        )])

    def forward(self, ids, feats, edge_dict, epo):

        x = feats
        # x = self.gcs[0](x, edge_dict)
        x = self.gcs[0](ids, x, edge_dict, epo)

        return x



