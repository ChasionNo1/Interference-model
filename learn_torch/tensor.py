# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/15 13:52
# @Author    :   Chasion
# Description:
import torch

a = torch.Tensor([1])
print(a)
x = a.item()
print(type(x))
