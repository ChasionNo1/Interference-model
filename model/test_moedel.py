from model.models import DHGNN_v3
from communication_model.load_data import load_data
import torch


feats, labels, idx_train, idx_val, idx_test, edge_dict = load_data()
n = feats.shape[0]
d = feats.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feats = torch.Tensor(feats).to(device)
model = DHGNN_v3(dim_in=d)
output = model(idx_train, feats, edge_dict, 6)
