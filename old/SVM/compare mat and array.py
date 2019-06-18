#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:34:39 2018

@author: suliang

矩阵和array的点乘叉乘的差别
"""


import numpy as np

ar1 = np.array([[1,2],
                [2,1]])
ar2 = np.array([[1,0],
                [1,0]])

mt1 = np.mat([[1,2],
              [2,1]])
mt2 = np.mat([[1,0],
              [1,0]])

#----------------数组array----------------
ar1*ar2                       # * 代表按位
np.multiply(ar1,ar2)          # multiply代表按位
np.dot(ar1,ar2)               # dot代表矩阵乘法

#----------------矩阵mat----------------
mt1*mt2                       # * 代表矩阵乘法
np.multiply(mt1, mt2)         # muliply 代表按位
np.dot(mt1,mt2)               # 代表矩阵乘法

# 总结下来就是： mulitiply为按位，dot为矩阵乘法
# 而最常用的*则被运算符重载为不同功能：
