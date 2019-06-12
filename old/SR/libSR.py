#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 14:54:47 2018

@author: suliang
"""

# lib for SR (softmax regression)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def BGD(data, labels, maxCycles, alpha):  # alpha为学习率
    k = len(set(labels))  # k为k个类别
    
    data = np.mat(data)
    labels = np.mat(labels).T
    x0 = np.ones((data.shape[0],1))
    data = np.hstack((x0, data))     # (135, 3) 第一列为1
    
    m, n = data.shape    
    thetas = np.mat(np.ones((n,k)))  # w初始化(n_feat, k_class), 每个样本对应一个类别就有一组参数，k个类别就有k组参数
    
    for i in range(maxCycles):
        edot = np.exp(data * thetas)  # data(135,3) * thetas(3,4) -> (135,4)4表示这个样本属于这4类的值
        rowsum = edot.sum(axis = 1)  # 计算每行求和
        rowsum = rowsum.repeat(k, axis=1)  # 横向赋值扩展成k列，每类一列备用
        
        p = - edot/rowsum  # (135,4)4表示每个样本属于一个类别的4个概率值 
                           # 计算softmax函数，等效于每个样本在每个类别的概率，k列
                           # 加负号是为了下一步算 1-p方便: 1-p变为1+(-p)
        for j in range(m):  # m 表示135个样本
            p[j, labels[j,0]] += 1  # 计算(1-p),只在该样本属于类的那列计算1-p
        # thetas = thetas + (alpha/m) * X.T * (1-p)
        thetas = thetas + (alpha / m) * data.T * p
        
    # cost计算    
    return thetas                                                 


def cost(err, labels):
    m = err.shape[0]
    sum_cost = 0.0
    pass


def loadDataSet(filename):
    df = pd.read_table(filename,header = None)
    data = df.iloc[:,0:2].values
    labels = df.iloc[:,-1].values

    return data, labels
    

def plotFitCurve(data, labels):    
    xmin = np.min(data[:,0])-1
    xmax = np.max(data[:,0])+1
    xi = np.arange(xmin, xmax, 0.1)
    
    colour = [c*30+60 for c in labels]
    plt.figure(figsize = (6,4), dpi=80)
    plt.scatter(data[:,0],data[:,1],c=colour)
    # theta0*1+theta1*x1+theta2*x2=0
    # x2 = (-theta0 - theta1*x1)/theta2
    #yi = (-thetas[0] - thetas[1] * xi) / thetas[2] 
    #plt.plot(xi,yi)
    

if __name__ == '__main__':
    inputfile = 'softInput.txt'
    data, labels = loadDataSet(inputfile)  # (135,2)  (135)
    
#    plotFitCurve(data,labels)
    thetas = BGD(data, labels, 200, 0.01)
    print('thetas = ', thetas)