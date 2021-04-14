# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/14 14:29
# @Author    :   Chasion
# Description:
import sympy

x = sympy.symbols('x')
a = sympy.solve(2 * x**2 - 4 * 100 * x + 10000, x)
print(a)