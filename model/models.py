from model.layers import *


class DHGNN_v1(nn.Module):
    """
    动态超图卷积使用图卷积输入层
    """
    def __init__(self, **kwargs):
        super(DHGNN_v1, self).__init__()
        self.dim_feat = kwargs['dim_feat']
        # 类别数
        self.n_categories = kwargs['n_categories']
        # 网络层数
        self.n_layer = kwargs['n_layer']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layer - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias'])]
            + [DHGLayer( dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            structured_neighbor=kwargs['k_structured'],
            nearest_neighbor=kwargs['k_nearest'],
            cluster_neighbor=kwargs['k_cluster'],
            wu_knn=kwargs['wu_knn'],
            wu_kmeans=kwargs['wu_kmeans'],
            wu_struct=kwargs['wu_struct'],
            n_cluster=kwargs['clusters'],
            n_center=kwargs['adjacent_centers'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layer)])

    def forward(self, **kwargs):
        ids = kwargs['ids']
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']
        G = kwargs['G`']
        ite = kwargs['ite']

        x = feats
        for i in range(self.n_layer):
            x = self.gcs[i](ids, x, edge_dict, G, ite)
        return x
