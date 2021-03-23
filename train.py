import argparse
import os
import torch
import copy
import time
import random
from config import get_config
from datasets import source_select
from torch import nn
import torch.optim as optim
from models import model_select
import sklearn
from sklearn import neighbors
import numpy as np
from utils.construct_hypergraph import _edge_dict_to_H, _generate_G_from_H


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0', help='version gpu id')
parser.add_argument('--model_version', default='DHGNN_v1', help='DHGNN model version, acceptable: DHGNN_v1, DHGNN_v2')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, fts, lbls, idx_train, idx_val, edge_dict, G, criterion, optimizer, scheduler, device, num_epoches=25, print_freq=500):
    """

    :param model:
    :param fts:
    :param lbls:
    :param idx_train:
    :param idx_val:
    :param edge_dict:
    :param G: G for input HGNN layer
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param device:
    :param num_epoches:
    :param print_freq:
    :return: best model on validation set
    """
    # 记录现在的时间
    since = time.time()
    # 计数器，每个epoch更新状态列表
    state_dict_update = 0
    # 使用cpu版本
    model = model.to(device)
    model_wts_best_val_acc = copy.deepcopy(model.state_dict())
    model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    loss_min = 100.0
    acc_epo = 0
    loss_epo = 0

    for epoch in range(num_epoches):
        epo = epoch

        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'epoch {epoch} / {num_epoches - 1}')

        # 每个epoch都有训练和验证环节
        for phase in ['train val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_val

            optimizer.zero_grap()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(ids=idx, feats=fts, edge_dict=edge_dict, G=G, ite=epo)
                loss = criterion(outputs, lbls[idx]) * len(idx)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss
            running_corrects += torch.sum(preds == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_wts_best_val_acc = copy.deepcopy(model.state_dict())
                acc_epo = epoch
                state_dict_update += 1

            if phase == 'val' and epoch_loss < loss_min:
                loss_min = epoch_loss
                model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())
                loss_epo = epoch
                state_dict_update += 1

            if epoch % print_freq == 0 and phase == 'val':
                print(f'best val acc: {best_acc:4f}, min val loss:{loss_min:4f}')
                print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'\nState dict updates {state_dict_update}')
    print(f'Best val Acc: {best_acc:4f}')

    return (model_wts_best_val_acc, acc_epo), (model_wts_lowest_val_loss, loss_epo)


def test(model, best_model_wts, fts, lbls, n_category, idx_test, edge_dict, G, device, test_time=1):
    """

    :param model:
    :param best_model_wts:
    :param tfs:
    :param lbls:
    :param n_category:
    :param idx_best:
    :param edge_dict:
    :param G:
    :param device:
    :param test_time:
    :return:
    """
    best_model_wts, epo = best_model_wts
    model = model.to(device)
    model.load_state_dict(best_model_wts)
    model.eval()

    running_corrects = 0.0
    outputs = torch.zeros(len(idx_test), n_category.to(device))

    for _ in range(test_time):
        with torch.no_grad():
            outputs += model(ids=idx_test, feats=fts, edge_dict=edge_dict, G=G, ite=epo)

    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds==lbls.data[idx_test])
    test_acc = running_corrects.double() / len(idx_test)

    print('*' * 20)
    print(f'test acc: {test_acc}  @epoch-{epo}')
    print('*' * 20)
    return test_acc, epo


def train_test_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = source_select(cfg)
    print(f'using {cfg["activate_dataset"]} dataset')
    fts, lbls, idx_train, idx_val, idx_test, n_category, _, edge_dict = source(cfg)
    H = _edge_dict_to_H(edge_dict)
    G = _generate_G_from_H(H)
    G = torch.Tensor(G).to(device)
    # 将数据转换为Tensor
    fts = torch.Tensor(fts).to(device)
    lbls = torch.Tensor(lbls).squeeze().long().to(device)

    model = model_select(cfg['model']) \
        (dim_feat=fts.size(1),
         n_categories=n_category,
         k_structured=cfg['k_structured'],
         k_nearest=cfg['k_nearest'],
         k_cluster=cfg['k_cluster'],
         wu_knn=cfg['wu_knn'],
         wu_kmeans=cfg['wu_kmeans'],
         wu_struct=cfg['wu_struct'],
         clusters=cfg['clusters'],
         adjacent_centers=cfg['adjacent_centers'],
         n_layers=cfg['n_layers'],
         layer_spec=cfg['layer_spec'],
         dropout_rate=cfg['drop_out'],
         has_bias=cfg['has_bias']
         )

    # 初始化模型
    state_dict = model.state_dict()
    for key in state_dict:
        if 'weight' in key:
            nn.init.xavier_uniform_(state_dict[key])
        elif 'bias' in key:
            state_dict[key] = state_dict[key].zero_()

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], eps=1e-20)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])

    criterion = torch.nn.NLLLoss()

    # 转换学习模式
    model_wts_best_val_acc, model_wts_lowest_val_loss \
        = train(model, fts, lbls, idx_train, idx_val, edge_dict, G, criterion, optimizer, schedular, device,
                cfg['max_epoch'], cfg['print_freq'])
    if idx_test is not None:
        print('**** Model of lowest val loss ****')
        test_acc_lvl, epo_lvl = test(model, model_wts_lowest_val_loss, fts, lbls, n_category, idx_test, edge_dict, G, device, cfg['test_time'])
        print('**** Model of best val acc ****')
        test_acc_bva, epo_bva = test(model, model_wts_best_val_acc, fts, lbls, n_category, idx_test, edge_dict, G, device, cfg['test_time'])
        return (test_acc_lvl, epo_lvl), (test_acc_bva, epo_bva)
    else:
        return None


if __name__ == '__main__':
    seed_num = 1000
    set_seed(seed_num)
    print('using random seedL:', seed_num)

    cfg = get_config('config/config.yaml')
    cfg['model'] = args.model_version
    train_test_model(cfg)






