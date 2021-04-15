# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/15 21:26
# @Author    :   Chasion
# Description:

def trans2bin(num):
    num = int(num*1000)/1000
    # print(num)
    integer_num = int(num)
    bin_1 = bin(integer_num)
    fra_num = num - integer_num
    bins = []
    for i in range(3):
        fra_num = fra_num * 2
        bins.append(1 if fra_num >= 1 else 0)
        fra_num = fra_num - int(fra_num)

    bin_list = list(bin_1[2:])
    bin_list = [int(x) for x in bin_list]
    fill_list = [0] * (11 - len(bin_list))
    bin_list = fill_list + bin_list

    bin_list = bin_list + bins


trans2bin(211.235654)
