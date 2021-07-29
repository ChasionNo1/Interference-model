# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/5/10 20:18
# @Author    :   Chasion
# Description:   格式化输出

s1 = '节点id       中心频率(MHz)          带宽(MHz)          接收机能量阈值(W)          发送功率(W)          X(km)            Y(km)'
cols_name = ['节点id', '中心频率(MHz)', '带宽(MHz)', '接收机能量阈值(W)', '发送功率(W)', 'X(km)', 'Y(km)']
name_len = [len(name) + 3 for name in cols_name]
print(name_len)
offset = 3
s2 = ''
for i in range(len(cols_name)):
    temp = cols_name[i] + ' ' * offset
    s2 += temp


print(s2)

