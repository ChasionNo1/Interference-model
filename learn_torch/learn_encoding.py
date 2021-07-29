# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/6/21 20:42
# @Author    :   Chasion
# Description:
"""
编码方式：
1、base64
"""
import base64

s = "20210621 熟视无睹".encode()
# 加密
res = base64.b64encode(s)
print(res.decode())
# 解密
res = base64.b64decode(res)
print(res.decode())
