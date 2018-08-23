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

def GA(data, labels, maxCycle, alpha):  # alpha为学习率
    data = np.mat(data)
    m, n = data.shape
    k = len(set(labels))  # k为k个类别
    
    thetas = np.mat(np.ones((n,k)))  # 初始化参数thetas, k个类别就是k列
    
    for i in range(maxCycles):
        err = np.exp(data * thetas)  # 计算mat点积
        rowsum = -err.sum(axis = 1)  # 计算每行求和
        rowsum = rowsum.repeat(k, axis=1)  # 横向赋值扩展
        
        err = err/rowsum
        for j in range(m):
            err[j, labels[j,0]] += 1
        # thetas = thetas + alpha*data.T*error
        thetas = thetas + (alpha / m) * data.T * err
        
        # -------------------------------
        value = np.exp(train_x * weights)  
        rowsum = value.sum(axis = 1)   # 横向求和
        rowsum = rowsum.repeat(k, axis = 1)  # 横向复制扩展
        err = - value / rowsum  #计算出每个样本属于每个类别的概率
        for j in range(numSamples):     
             err[j, train_y[j]] += 1
        weights = weights + (alpha / numSamples) * (train_x.T * err)
        #-------------------------------
                                                      

def BGA(data, labels, alpha=0.001):  # batch gradient ascent 批量梯度上升
# alpha 为学习率
# 内定最大循环次数为500次
    data = np.mat(data)
    x0 = np.ones((data.shape[0],1)) # 增加首列全1，用于匹配thetas首列的偏置b
    data = np.hstack((x0, data))
    
    maxCycles = 500
    labels = np.mat(labels).T
    m, n = data.shape
    thetas = np.ones([n,1]) # 新建thetas,默认为全1列表
        
    for i in range(maxCycles): # 每次都使用整个数据集data计算一列hxi,得到一列thetas
        hxi = sigmoid(data * (thetas))  # mat格式下*代表叉乘, sigmoid()可以计算矢量
        error = labels - hxi            # labels, hxi, error都是一列矢量
        thetas += alpha * data.T * error   # 整个data更新一组thetas
        
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
    data, labels = loadDataSet(inputfile)
    
    plotFitCurve(data,labels)