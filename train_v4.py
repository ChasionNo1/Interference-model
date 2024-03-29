# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/25 10:39
# @Author    :   Chasion
# Description:
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/11 20:01
# @Author    :   Chasion
# Description:
import os
import torch
import copy
import time
import random
from communication_model.load_data import load_data
from torch import nn
import torch.optim as optim
import numpy as np
from model.models import DHGNN_V4
from matplotlib.pylab import plt
from communication_model.load_prediction_data import load_data2
from Data.load_cora import load_citation_data

# load cora数据集
"""
更改说明：
标签文件，不需要再onehot
损失函数，更换
全连接层的输出维度：7
softmax,以及使用图卷积作为输入，
"""


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, feats, labels, idx_train, idx_val, edge_dict, criterion, optimizer, scheduler, device, num_epoches, print_freq=500):
    """

    :param model:
    :param feats:
    :param labels:
    :param idx_train:
    :param idx_val:
    :param edge_dict:
    :param adj:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param device:
    :param num_epoches:
    :param print_freq:
    :return:
    """
    # 记录开始训练的时间
    since = time.time()
    # 模型更新次数计数器
    state_dict_update_num = 0
    # 使用cpu计算
    model = model.to(device)
    # 最佳性能
    model_wts_best_val_acc = copy.deepcopy(model.state_dict())
    model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())

    # 初始化指标
    best_acc = 0.0
    loss_min = 100.0
    acc_epoch = 0
    loss_epoch = 0

    for epoch in range(num_epoches):

        print('-' * 40)
        print(f'epoch {epoch}/{num_epoches-1}')

        # 每个epoch都有训练和验证
        for phase in ['train', 'val']:

            if phase == 'train':
                # 学习率衰减
                scheduler.step()
                model.train()
            else:
                model.eval()
            # 运行时的loss和acc
            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_val
            # 梯度置0
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                # 前向传播
                output = model(ids=idx, feats=feats, edge_dict=edge_dict, epo=epoch)
                # LOSS
                loss = criterion(output, labels[idx]) * len(idx)

                _, preds = torch.max(output, 1)

                if phase == 'train':
                    # 训练的时候，需要反向传播优化参数，验证时不需要
                    loss.backward()
                    optimizer.step()

            running_loss += loss
            # _, labels_max = torch.max(labels.data[idx], 1)
            running_corrects += torch.sum(preds == labels.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)
            if phase == 'train':
                train_loss_list.append(epoch_loss.detach().numpy())
                train_acc_list.append(epoch_acc.detach().numpy())

            if phase == 'val':
                val_loss_list.append(epoch_loss.detach().numpy())
                val_acc_list.append(epoch_acc.detach().numpy())
            # if epoch % print_freq == 0:
            print(f'{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}')

            # 拷贝最优模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_wts_best_val_acc = copy.deepcopy(model.state_dict())

                acc_epoch = epoch
                state_dict_update_num += 1

            if phase == 'val' and epoch_loss < loss_min:
                loss_min = epoch_loss
                model_wts_best_val_acc = copy.deepcopy(model.state_dict())

                loss_epoch = epoch
                state_dict_update_num += 1
            # epoch % print_freq == 0 and
            if phase == 'val':
                print(f'Best val Acc: {best_acc:4f}, Min val loss: {loss_min:4f}')
                print('-' * 40)
    token = time.time()
    token = int(token)
    # 模型保存到本地
    torch.save({'epoch': acc_epoch,
                'model_state_dict': model_wts_best_val_acc,
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'weights/model_wts_best_val_acc{}.pkl'.format(token))
    torch.save({'epoch': loss_epoch,
                'model_state_dict': model_wts_lowest_val_loss,
                'optimizer_state_dict': optimizer.state_dict()
                }, 'weights/model_wts_lowest_val_loss{}.pkl'.format(token))
    # 训练结束
    # 计算用时
    total_time = time.time() - since
    print('-' * 40)
    print(f'训练用时：{total_time}s')
    print(f'模型更新次数：{state_dict_update_num}')
    print(f'最佳验证正确率：{best_acc:.4f}')
    print('\n')

    return (model_wts_best_val_acc, acc_epoch), (model_wts_lowest_val_loss, loss_epoch)


def test(model, best_model_wts, feats, labels, idx_test, edge_dict, device, n_categories=7, test_time=1):

    best_model_wts, epo = best_model_wts
    model = model.to(device)
    model.load_state_dict(best_model_wts)
    model.eval()

    running_corrects = 0.0
    outputs = torch.zeros(len(idx_test), n_categories).to(device)

    for _ in range(test_time):
        with torch.no_grad():
            outputs += model(ids=idx_test, feats=feats, edge_dict=edge_dict, epo=epo)

    _, preds = torch.max(outputs, 1)

    running_corrects += torch.sum(preds == labels.data[idx_test])
    test_acc = running_corrects.double() / len(idx_test)

    print('*' * 20)
    print(f'test acc:{test_acc}')
    print('*' * 20)
    return test_acc


def train_and_test_model():
    """
    这部分不适用配置文件导入参数
    :return:
    """
    # 使用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_data

    path = r'D:\graph_code\Interference-model\Data\\cora'
    cfg = {'activate_dataset': 'cora',
           'citation_root': path,
           'add_self_loop': True,
           }
    feats, labels, idx_train, idx_val, idx_test, n_category, edge_dict, _ = load_citation_data(cfg)
    # feats, labels, idx_train, idx_val, idx_test, edge_dict = load_data()
    feats = torch.Tensor(feats).to(device)
    # one_hot = np.eye(len(labels), 2)
    # labels = one_hot[labels]
    labels = torch.Tensor(labels).squeeze().long().to(device)
    print(labels)
    model = DHGNN_V4(
        dim_feats=feats.size(1)
    )

    state_dict = model.state_dict()
    for key in state_dict:
        if 'weight' in key:
            nn.init.xavier_uniform_(state_dict[key])
        if 'bias' in key:
            state_dict[key] = state_dict[key].zero_()

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005, eps=1e-20)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
    criterion = torch.nn.NLLLoss()
    epoch_num = 24
    # model, feats, labels, idx_train, idx_val, edge_dict, adj, criterion, optimizer, scheduler, device, num_epoches=25, print_freq=500
    model_wts_best_val_acc, model_wts_lowest_val_loss = train(model, feats, labels, idx_train, idx_val, edge_dict, criterion, optimizer, scheduler, device, epoch_num)

    # test
    # pre_feats, pre_labels, pre_idx, pre_edge_dict = load_data2()
    # pre_feats = torch.Tensor(pre_feats)
    # pre_one_hot = np.eye(len(pre_labels), 2)
    # pre_labels = pre_one_hot[pre_labels]
    # pre_labels = torch.Tensor(pre_labels).float().to(device)
    #
    # acc1 = test(model, model_wts_best_val_acc, pre_feats, pre_labels, pre_idx, pre_edge_dict, device)
    # print('acc1', acc1)
    if idx_test is not None:
        print('test part')
        acc = test(model, model_wts_best_val_acc, feats, labels, idx_test, edge_dict, device)
        acc = str(acc.item())
        with open('acc.txt', 'a')as f:
            f.write(acc)
            f.write('\n')


if __name__ == '__main__':
    seed_num = 1000
    setup_seed(seed_num)
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    train_and_test_model()

    plt.plot(train_loss_list, label='train_loss')
    plt.plot(train_acc_list, label='train_acc')
    plt.plot(val_loss_list, label='val_loss')
    plt.plot(val_acc_list, label='val_acc')
    plt.legend()
    token = time.time()
    plt.savefig('result/result_{}'.format(int(token)))
    plt.show()










