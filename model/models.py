from model.layers02 import *


class DHGNN(nn.Module):
    """
    动态超图神经网络：采用图卷积作为输入层
    """
    def __init__(self, **kwargs):
        super(DHGNN, self).__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        self.layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + self.layer_spec
        self.dims_out = self.layer_spec + [self.n_categories]
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=self.activations[0],
            has_bias=kwargs['has_bias'])
            + [DHGLayer(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=self.activations[i],
            has_bias=kwargs['has_bias'])] for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        ids = kwargs['ids']
        feats = kwargs['feats']
        adj = kwargs['adj']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i in range(self.n_layers):
            x = self.gcs[i](ids, x, edge_dict, adj)
        return x
