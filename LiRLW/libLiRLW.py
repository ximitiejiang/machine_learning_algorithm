#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:26:53 2018

@author: suliang
"""

# LiRLW = linear regression with locally weighted 局部加权线性回归

import numpy as np
import pandas as pd

def lirRegreLW(x_test, data, label, k = 1.0):
    data = np.mat(data)
    label = np.mat(label).T
    
    m = data.shape[0]
    y_test = np.zeros(m)
    weights = np.mat(np.eye(m))  # 新建对每个样本的权重矩阵
    for i in range(m): # 外层循环，取出每一个样本xi
        
        for j in range(m):  # 内层循环，针对每一个样本xi，更新一组theta
            diffMat = x_test[i] - data[j,:]  
            theta[j,j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))
    
        xTx_w = data.T * (weights * data)
        if np.linalg.det(xTx) == 0:
            print('matrix can not inverse')
            return    
        theta = xTx_w.I * (data.T * (weights *label))
    
    y_test = x_test * theta
    
    return y_test


    
 