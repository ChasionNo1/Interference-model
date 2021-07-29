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
    # figsize=(width, height)
    midu = 2
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(8, 6), dpi=100)

    axs[0, 0].plot(data0['train_loss'], 'bo-.', label='Random', markevery=midu)
    axs[0, 0].plot(data1['train_loss'], 'r^-', label='ASHG', markevery=midu)
    axs[0, 0].set(title='train loss', ylabel='Loss')
    axs[0, 0].grid(linestyle='-.')

    axs[0, 1].plot(data0['val_loss'], 'bo-.', label='Random', markevery=midu)
    axs[0, 1].plot(data1['val_loss'], 'r^-', label='ASHG', markevery=midu)

    axs[0, 1].set(title='val_loss', ylabel='Loss')
    # axs[0, 1].legend(loc=2, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.)
    axs[0, 1].grid(linestyle='-.')
    plt.xlim(0, 35)
    plt.ylim(0.2, 1)

    axs[1, 0].plot(data0['train_acc'], 'bo-.', label='Random', markevery=midu)
    axs[1, 0].plot(data1['train_acc'], 'r^-', label='ASHG', markevery=midu)
    axs[1, 0].set(title='train_accuracy', xlabel='epoch', ylabel='Accuracy')
    axs[1, 0].grid(linestyle='-.')

    axs[1, 1].plot(data0['val_acc'], 'bo-.', label='Random', markevery=midu)
    axs[1, 1].plot(data1['val_acc'], 'r^-', label='ASHG',markevery=midu)

    axs[1, 1].set(title='val_accuracy', xlabel='epoch', ylabel='Accuracy')
    axs[1, 1].legend(loc=4)
    axs[1, 1].grid(linestyle='-.')

    plt.tight_layout()
    plt.savefig('20210722.svg')
    plt.show()


if __name__ == '__main__':
    path = r'D:\graph_code\Interference-model\result\data\722result.txt'
    load_data()
