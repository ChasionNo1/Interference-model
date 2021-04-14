from model.models import DHGNN_v2
from Data.load_data import load_data
import torch


feats, adj, labels, idx_train, idx_val, idx_test, edge_dict = load_data()
n = feats.shape[0]
d = feats.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feats = torch.Tensor(feats).to(device)
model = DHGNN_v2(dim_in=d)
output = model(idx_train, feats, edge_dict, adj, 6)
