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
import random
import pickle as plk


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_files(content, path):
    """
    序列化数据
    :param path:
    :return:
    """
    with open(path, 'wb') as f:
        plk.dump(content, f)


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
    # 输出每一行的最大值
    _, preds = torch.max(outputs, 1)
    _, labels_max = torch.max(labels.data[idx], 1)
    # 计算指标，
    # print(preds)
    # TP
    preds0 = torch.where(preds == 0)

    label_tp0 = torch.where(labels_max == 0)
    count0 = torch.where(preds[label_tp0[0]] == 0)
    tp0 = count0[0].size()[0]
    fn0 = label_tp0[0].size(0) - tp0
    precision = tp0 / (tp0 + fn0)
    fp0 = preds0[0].size()[0] - label_tp0[0].size(0)
    # print(precision)
    # recall = tp/tp+fp
    recall = tp0 / (tp0 + fp0)
    # print(recall)
    # f1 = 2*precision*recall/ (precision+recall)
    f1 = 2*precision*recall/(precision + recall)
    # print(f1)
    label_tp1 = torch.where(labels_max == 1)
    running_corrects += torch.sum(preds == labels_max)
    # FN =
    # print(TP)
    test_acc = running_corrects.double() / len(idx)

    evaluation = {'accuracy': test_acc.item(),
                  'precision': precision,
                  'recall': recall,
                  'f1': f1
                  }
    # write_files(evaluation, 'evaluation.pkl')
    with open('evaluation.txt', 'a')as f:
        f.write(str(evaluation) + '\n')
    print('预测正确率：', test_acc.item())


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats, labels, idx_train, idx_val, idx, edge_dice = load_data2()
    # idx = [i for i in range(400, 420)]
    # print(idx)
    setup_seed(100)
    feats = torch.Tensor(feats)
    one_hot = np.eye(len(labels), 2)
    labels = one_hot[labels]
    labels = torch.Tensor(labels).float().to(device)
    model = DHGNN_v3(
        dim_feats=feats.size()[1]
    )
    path = 'weights/model_wts_best_val_acc.pkl'
    predict(model, path, feats, labels, idx, edge_dice, device)


