import torch
from torch import nn
import pandas as pd


def cos_dis(X):
        """
        cosine distance   余弦距离 相似度计算
        1、将数据映射为高维空间中的点（向量）
        2、计算向量之间的余弦值
        3、取值范围【-1，+1】，越趋近于1代表越相似，0代表正交
        :param X: (N, d)
        :return: (N, N)
        """
        # 计算X的归一化，torch.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12, out=None)
        X = nn.functional.normalize(X)
        # 0维和1维调换位置
        XT = X.transpose(0, 1)
        # 因为先进行了归一化，在求解余弦值的时候，分母本来就是1了，所以只需要计算矩阵a和b的乘积
        return torch.matmul(X, XT)


def sample_ids(ids, k):
    """
    sample `k` indexes from ids, must sample the centroid node itself  ID的索引，必须对质心顶点本身进行采样
    :param ids: indexes sampled from  从中采样的索引
    :param k: number of samples   采样数
    :return: sampled indexes   采样索引
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k - 1, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    sampled_ids.append(ids[-1])  # must sample the centroid node itself
    return sampled_ids


def sample_ids_v2(ids, k):
    """
    purely sample `k` indexes from ids
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    return sampled_ids