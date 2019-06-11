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
    x = df.iloc[:,0:2].values
    y = df.iloc[:,-1].values
    return x, y


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


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
        prob = data * (thetas)  # mat格式下*代表按位相乘后累加, sigmoid()可以计算矢量
        hxi = sigmoid(prob)
        error = labels - hxi            # labels, hxi, error都是一列矢量
        thetas += alpha * data.T * error   # 整个data更新一组thetas = theta + (100,3)*(100,1)
        
    return thetas


def BGA2(data, labels, alpha=0.001):  # 批量提取上升，并保留每次theta用于可视化梯度变化
# BGA2用来查看回归曲线的变化情况
# 保存每次循环theta的值
    data = np.mat(data)
    x0 = np.ones((data.shape[0],1))
    data = np.hstack((x0, data))
    
    maxCycles = 500
    labels = np.mat(labels).T
    m, n = data.shape
    thetas = np.ones([n,maxCycles]) # 修改：循环多少次就存多少次thetas
        
    for i in range(maxCycles):
        hxi = sigmoid(np.dot(data, thetas[:,i].reshape(-1,1)))  # 修改：
        error = labels - hxi
        # numpy的奇葩更新方式：矩阵某列(3,) = 行向量(1,3), 不能等于列向量，
        # 否则报错，所以加了一个转秩
        # 同时累加格式+=不支持，只能改为thetas = thetas + xxx...
        # 同时为了累加，每次把更新值填充i之后的所有列。
        for j in range(i, maxCycles):
            thetas[:,j] = thetas[:,j] + np.array(alpha * data.T * error).T # 修改：
        
    return thetas



def SGA(data, labels, alpha=0.001):  # 随机梯度上升: 效果跟BGA类似，但占用更少资源
    m, n = np.shape(data)
    thetas = np.ones(n)
    
    data = np.mat(data)
    x0 = np.ones((data.shape[0],1)) # 增加首列全1，用于匹配thetas首列的偏置b
    data = np.hstack((x0, data))
    
    for i in range(m):   # 每次只用1行样本计算一个hxi/error,得到一列thetas的更新
        hxi = sigmoid(sum(data[i]*thetas)) # hxi为数值
        error = labels[i] - hxi
        thetas = thetas + alpha * error * data[i] # 一行样本更新一组thetas
    return thetas


def plotFitCurve(data, labels, thetas):    
    xmin = np.min(data[:,0])-1
    xmax = np.max(data[:,0])+1
    xi = np.arange(xmin, xmax, 0.1)
    
    colour = [c*30+60 for c in labels]
    plt.figure(figsize = (6,4), dpi=80)
    plt.scatter(data[:,0],data[:,1],c=colour)
    # theta0*1+theta1*x1+theta2*x2=0
    # x2 = (-theta0 - theta1*x1)/theta2
    yi = (-thetas[0] - thetas[1] * xi) / thetas[2] 
    plt.plot(xi,yi)

    
def plotFitCurve2(data, labels, thetas):
    
    xmin = np.min(data[:,0])-1
    xmax = np.max(data[:,0])+1
    x = np.arange(xmin, xmax, 0.1)
    
    colour = [c*30+60 for c in labels]
    plt.figure(figsize = (6,4), dpi=80)
    plt.scatter(data[:,0],data[:,1],c=colour)
    y = []
    for i in range(5):
        y.append((-thetas[0,100*i] - thetas[1,100*i] * x) / thetas[2,120*i]) # 
        
        plt.plot(x,y[i])


def classify(x, thetas):  # 
    x = np.hstack((1,x))  # 在BGA训练时x需要加一列1，在新样本分类数据之前也需要加一列1
    p = sigmoid(sum(np.dot(x,thetas)))  # 计算逻辑函数的值，大于0.5代表概率大于50%判定为1
    if p >= 0.5:
        return 1.0
    else: return 0.0


def classify_OvsO():  # 一对一：每一个类跟其他每类建立一个分类器(共k!个)，最后投票正确类肯定多1票
    # 逻辑回归默认只能进行二分类问题，如果要做多分类，比如手写数字识别，需要修改
    # 一对一: 需要的模型多，但每个模型训练样本少，适合巨量样本
    # 如果分类互斥，用softmax；如果分类可同时存在，用OvsO或OvsR
    pass


def classify_OvsR():  # 一对剩余：每一个类跟剩余类建立一个分类器(共k个)，最后投票唯一不同的票就是正确类
    # 一对剩余：需要的模型少，但每个模型训练样本数多
    # 一般来说一对剩余相对高效点。但总体差别不大。
    pass

#----------------------------------------------------------------
# libLoR测试函数运行方式： import libLoR; libLoR.test()
def test(sample):  # 简单样本实例：sample = [0,1]这种list类型
    x, y = loadDataSet('testSet.txt')  
    thetas = BGA(x, y)       # 在训练数据x,y中得到模型参数thetas
    plotFitCurve(x, y, thetas)      # 对模型参数和回归曲线进行可视化
    
    result = classify(sample, thetas)
    print('the sample {} belong to class {}'.format(sample, result))
    return thetas, result   # 根据训练数据，分界线以上为0，分界线以下为1

def test_1(sample):  # 
    x, y = loadDataSet('testSet.txt')  
    thetas = BGA2(x, y)         # 改用记录每次thetas的BGA
    plotFitCurve2(x, y, thetas) # 改用可以画出每100个循环一根的分界线
    
    result = classify(sample, thetas)
    print('the sample {} belong to class {}'.format(sample, result))
    return thetas, result   # 根据训练数据，分界线以上为0，分界线以下为1

def test_3(sample):  # 改用SGA
    x, y = loadDataSet('testSet.txt')  
    thetas = SGA(x, y)       # 在训练数据x,y中得到模型参数thetas
    plotFitCurve(x, y, thetas)      # 对模型参数和回归曲线进行可视化
    
    result = classify(sample, thetas)
    print('the sample {} belong to class {}'.format(sample, result))
    return thetas, result   # 根据训练数据，分界线以上为0，分界线以下为1        
    
def test_Colic():  # 疝气病实例
    pass
    

thetas, result = test([0,-100])    # BGA
#thetas, result = test_1([0,-100])  # BGA2
#thetas, result = test_3([0,-100])   # SGA

print(thetas)


