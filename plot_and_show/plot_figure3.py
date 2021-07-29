# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/5/15 16:38
# @Author    :   Chasion
# Description:   knn and kmeans hyperedge size accuracy
import numpy as np
import matplotlib.pyplot as plt


path = r'D:\graph_code\Interference-model\acc.txt'
with open(path, 'r')as f:
    data = f.readlines()
data = [float(x) for x in data]
x1 = np.array([5, 10, 15, 20, 25])
print(x1)
y1 = data[:5]
y2 = data[5:]
plt.plot(x1, y1, c='b', label='only knn', marker='')
plt.plot(x1, y2, c='r', label='only kmeans')
plt.ylabel('Accuracy')
plt.xlabel('Hyperedge Size')
plt.ylim(0.5, 1.2)
plt.legend()
plt.show()
