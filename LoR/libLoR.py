#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:53:38 2018

@author: suliang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadDataSet(filename):
    df = pd.read_table(filename,header = None)
    X = df.iloc[:,0:2].values
    y = df.iloc[:,-1].values
    return X, y


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def BGA(data, labels, alpha=0.001):
# alpha 为学习率
# 内定最大循环次数为500次
    data = np.mat(data)
    x0 = np.ones((data.shape[0],1))
    data = np.hstack((x0, data))
    
    maxCycles = 500
    labels = np.mat(labels).T
    m, n = data.shape
    thetas = np.ones([n,1]) # 循环多少次就存多少次thetas，看看每次出来曲线长什么样子
        
    for i in range(maxCycles):
        hxi = sigmoid(data * (thetas))  # mat格式下*代表叉乘
        error = labels - hxi
        thetas += alpha * data.T * error   #???
        
    return thetas


def BGA2(data, labels, alpha=0.001):
# BGA2用来查看回归曲线的变化情况
# 保存每次循环theta的值
    data = np.mat(data)
    x0 = np.ones((data.shape[0],1))
    data = np.hstack((x0, data))
    
    maxCycles = 500
    labels = np.mat(labels).T
    m, n = data.shape
    thetas = np.ones([n,maxCycles]) # 循环多少次就存多少次thetas
        
    for i in range(maxCycles):
        hxi = sigmoid(data * (thetas[:,i].reshape(-1,1)))  # mat格式下*代表叉乘
        error = labels - hxi
        thetas[:,i] += alpha * data.T * error   #???
        
    return thetas


def SGA(data, labels, alpha):
    pass
    

def BGD():
    pass


def plotFitCurve(data, labels, thetas):
    
    xmin = np.min(data[:,0])-1
    xmax = np.max(data[:,0])+1
    x = np.arange(xmin, xmax, 0.1)
    
    colour = [c*30+60 for c in labels]
    plt.figure(figsize = (6,4), dpi=80)
    plt.scatter(data[:,0],data[:,1],c=colour)
    y = (-thetas[0] - thetas[1] * x) / thetas[2] # 
    plt.plot(x,y)
    
def plotFitCurve2(data, labels, thetas):
    
    xmin = np.min(data[:,0])-1
    xmax = np.max(data[:,0])+1
    x = np.arange(xmin, xmax, 0.1)
    
    colour = [c*30+60 for c in labels]
    plt.figure(figsize = (6,4), dpi=80)
    plt.scatter(data[:,0],data[:,1],c=colour)
    y = []
    for i in range(5):
        y[i] = (-thetas[0,i*100] - thetas[1,i*100] * x) / thetas[2,i*100] # 
        plt.plot(x,y[i])

# -------main-----------
data, labels = loadDataSet('testSet.txt')  # X, y都是np.array

thetas = BGA(data, labels, 0.001)

plotFitCurve(data, labels, thetas)