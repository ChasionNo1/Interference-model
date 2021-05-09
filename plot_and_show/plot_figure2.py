# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/5/7 10:55
# @Author    :   Chasion
# Description:   超边大小对准确率的影响
import matplotlib.pyplot as plt
import numpy as np


def plot_result():
    path = r'D:\graph_code\Interference-model\result\data\25val_loss.txt'
    with open(path, 'r')as f:
        data = f.readlines()
    data0 = eval(data[0])
    data1 = eval(data[1])
    data2 = eval(data[2])

    data3 = eval(data[3])
    data4 = eval(data[4])
    data5 = eval(data[5])

    data6 = eval(data[6])
    data7 = eval(data[7])
    data8 = eval(data[8])
    data9 = eval(data[9])
    data10 = eval(data[10])
    data11 = eval(data[11])
    data12 = eval(data[12])

    data13 = eval(data[13])
    data14 = eval(data[14])
    data15 = eval(data[15])
    data16 = eval(data[16])



    # 最下面那个
    ax1 = plt.subplot(221)
    ax1.set_title('')
    # 绘制 10% 15% 20% 25% 下超边大小的准确率
    ax1.plot(data0, label='5')
    ax1.plot(data1, label='10')
    ax1.plot(data2, label='15')
    ax1.set_ylim(0, 1.25)
    ax1.set_title('10% knn train loss')

    ax1.legend()
    ax2 = plt.subplot(222)
    ax2.plot(data3, label='5')
    ax2.plot(data4, label='10')
    ax2.plot(data5, label='15')
    ax2.set_ylim(0, 1.25)
    ax2.set_title('10% kmeans train loss')
    ax2.legend()

    ax3 = plt.subplot(223)
    # ax3.plot(data6, label='18')
    ax3.plot(data7, label='19')
    ax3.plot(data8, label='20')
    ax3.plot(data9, label='21')
    ax3.plot(data10, label='22')
    # ax3.plot(data11, label='23')
    # ax3.plot(data12, label='24')
    ax3.set_ylim(0, 1.25)
    ax3.set_title('10% structure train loss')
    ax3.legend()
    #
    ax4 = plt.subplot(224)
    ax4.plot(data13, label='no kmeans')
    ax4.plot(data14, label='no structure')
    ax4.plot(data15, label='no knn')
    ax4.plot(data16, label='all')
    ax4.set_ylim(0, 1.25)
    ax4.set_title('10% val loss')
    ax4.legend()


    plt.show()


if __name__ == '__main__':
    plot_result()
