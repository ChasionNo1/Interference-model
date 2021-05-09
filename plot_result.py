# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/21 14:26
# @Author    :   Chasion
# Description:   画图
import matplotlib.pyplot as plt
import numpy as np
import pickle as plk
import matplotlib.font_manager
from matplotlib.pyplot import MultipleLocator


zhfont1 = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Bold.otf")


def load_data():
    with open('evaluation.txt', 'r')as f:
        data = f.readlines()
    data1 = eval(data[0])
    data2 = eval(data[1])
    arr1 = list(data1.values())
    arr2 = list(data2.values())
    acc = [0.85, 0.91]
    p = [0.9642857142857143, 0.9880952380952381]
    r = [0.9, 0.9222222222222223]
    f1 = [0.9310344827586207, 0.9540229885057472]
    return arr1, arr2


def plot_bar():
    data1, data2 = load_data()
    # data1 = [0.85, 0.9642857142857143, 0.9, 0.9310344827586207]
    # data2 = [0.91, 0.9880952380952381, 0.9222222222222223, 0.9540229885057472]
    name_list = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(name_list))
    # print(x)
    total_width, n = 0.6, 2
    width = total_width/n
    x = x - (total_width - width) / 2
    # 设置大小
    plt.figure(figsize=(6, 6))
    plt.bar(x, data1, width=width, label='ALTH', hatch='//', color='w', edgecolor="k")
    plt.bar(x + width, data2, width=width, label='Random', hatch='...', color='w', edgecolor="k")
    plt.title('results of metrics')
    plt.xticks([0, 1, 2, 3], name_list)
    # plt.ylim(0, 1.5)
    plt.legend(loc=2, bbox_to_anchor=(0, 1.11), borderaxespad=0.)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle='-.')
    plt.show()


if __name__ == '__main__':
    plot_bar()
