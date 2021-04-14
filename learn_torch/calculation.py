# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/13 18:30
# @Author    :   Chasion
# Description:
import math
import numpy as np
import torch


a = 20 * math.log10(2) + 20 * math.log10(400) + 32.4
print(a)

b = 10 * np.log10(20000)
print(b)

c = b - a + 4
print(c)

d = 10 * math.log10(4e-4)
print(d)

np_array = np.random.randint(1, 10, size=(4, 3))
t = torch.Tensor(np_array)
print(t)
