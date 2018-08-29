#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:26:53 2018

@author: suliang
"""

# LiRLW = linear regression with locally weighted 局部加权线性回归

import numpy as np
import pandas as pd



def loadDataSet(filename):
    df = pd.read_table(filename,header = None)
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values     # 假定所有数据最后一列都是label
    return x, y


def LWlinearRegres(x_test, data, label, k = 1.0):
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


def plotRegresCurve(data, label, theta):
    import matplotlib.pyplot as plt
    
    x_copy = data.copy()  # 复制一个数据不能简单的变量名复制，这样指针会指向同一地方
    x_copy.sort(axis = 0)
    y_regres = x_copy * theta  # 计算预测值 y = theta * x
    
    fig = plt.figure()
    plt.scatter(data[:,1], label)
    plt.plot(x_copy[:,1],y_regres, 'r--')


def test():
    filename = 'ex0.txt'
    data, label = loadDataSet(filename)
    y_test = LWlinearRegres(x_test, data, label, k=1.0)
    plotRegresCurve(data, label, theta)   
 