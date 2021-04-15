# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/15 18:33
# @Author    :   Chasion
# Description: 预测文件
import torch
from model.models import DHGNN_v3
from communication_model.load_prediction_data import load_data2
from communication_model.load_data import load_data
import numpy as np


def predict(model, path, feats, labels, idx, edge_dict, device, n_categories=2, test_time=1):
    model_ckpt = torch.load(path)
    model.load_state_dict(model_ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    running_corrects = 0.0
    outputs = torch.zeros(len(idx), n_categories).to(device)

    for _ in range(test_time):
        with torch.no_grad():
            outputs += model(ids=idx, feats=feats, edge_dict=edge_dict, epo=model_ckpt['epoch'])

    _, preds = torch.max(outputs, 1)
    _, labels_max = torch.max(labels.data[idx], 1)
    running_corrects += torch.sum(preds == labels_max)
    test_acc = running_corrects.double() / len(idx)

    print('预测正确率：', test_acc.item())


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats, labels, idx_train, idx_val, _, edge_dice = load_data()
    idx = [i for i in range(400, 420)]
    print(idx)
    feats = torch.Tensor(feats)
    one_hot = np.eye(len(labels), 2)
    labels = one_hot[labels]
    labels = torch.Tensor(labels).float().to(device)
    model = DHGNN_v3(
        dim_feats=feats.size()[1]
    )
    path = 'weights/model_wts_best_val_acc.pkl'
    predict(model, path, feats, labels, idx, edge_dice, device)


