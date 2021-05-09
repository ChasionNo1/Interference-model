# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/5/6 15:41
# @Author    :   Chasion
# Description:   对比三种方式的结果
import matplotlib.pyplot as plt
import numpy as np


def load_data():

    with open(path, 'r') as f:
        data = f.readlines()
    data0 = eval(data[0])
    data1 = eval(data[1])
    data2 = eval(data[2])
    # figsize=(width, height)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[8, 5], dpi=100)

    # axs[0, 0].plot(data0['train_loss'])
    # axs[0, 0].plot(data1['train_loss'])
    # axs[0, 0].plot(data2['train_loss'])

    axs[0].plot(data0['val_loss'], 'bo-.', label='Random')
    axs[0].plot(data1['val_loss'], 'r^-', label='AHTL')
    axs[0].plot(data2['val_loss'], 'gd--', label='ALTH')
    axs[0].set(title='val_loss', ylabel='Loss')
    axs[0].grid(linestyle='-.')
    plt.xlim(0, 35)
    plt.ylim(0.5, 1)
    # axs[1, 0].plot(data0['train_acc'])
    # axs[1, 0].plot(data1['train_acc'])
    # axs[1, 0].plot(data2['train_acc'])

    axs[1].plot(data0['val_acc'], 'bo-.', label='Random')
    axs[1].plot(data1['val_acc'], 'r^-', label='AHTL')
    axs[1].plot(data2['val_acc'], 'gd--', label='ALTH')
    axs[1].set(title='val_accuracy', xlabel='epoch', ylabel='Accuracy')
    axs[0].legend(loc=2, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.)
    axs[1].grid(linestyle='-.')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = r'D:\graph_code\Interference-model\result\data\result.txt'
    load_data()
