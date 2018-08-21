#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:34:39 2018

@author: suliang

矩阵和array的点乘叉乘的差别
"""


import numpy as np

ar1 = np.array([[1,2],[2,1]])
ar2 = np.array([[1,0],[2,1]])

mt1 = np.mat(ar1)
mt2 = np.mat(ar2)


ar_star = ar1*ar2
ar_multi = np.multiply(ar1,ar2)
ar_dot = np.dot(ar1,ar2)

mt_star = mt1*mt2
mt_multi = np.multiply(mt1, mt2)
mt_dot = np.dot(mt1,mt2)