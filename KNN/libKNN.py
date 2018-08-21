#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:56:52 2018

@author: suliang

lib_KNN库文件说明:
    - clf_point(inX, dataSet, labels, k), KNN算法
    - autonorm(data), 归一化算法
    - createDataSet(), 一个简单的测试数据生成函数
    - gaussian(dist, a=1, b=0, c=0.3), 对距离进行加权方法之高斯加权

        
"""

import numpy as np
import operator
import pandas as pd

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) # 采用最常规的array格式
    labels = np.array([0, 0, 1, 1]).reshape(-1,1)  # 采用最常规的列向量array格式
    return group, labels

# clf_point分类器，用于一个点的分类
def clf_point(inX, dataSet, labels, k):       # clf为分类器 
    # inX为待分类数据比如2个特征的[0,0]，也可以是n个特征比如[0.2,0.3,0.5,1.3]
    # dataset为X_train, label为y_train, k为近邻个数
    row = dataSet.shape[0]       # 读取X行数
    diffMat = np.tile(inX, (row,1)) - dataSet  # 待分类数据升维，与样本相减
    sqDiffMat = diffMat**2     # 平方
    distance = (np.sum(sqDiffMat, axis=1))**0.5    #求和后开方得到距离，为标量
    sort_dist = np.argsort(distance)  # 把距离升序排序：返回index
    
    classcount = {}   # 定义字典存储: {label: 个数} 用来完成KNN投票决定分类
    for i in range(k):
        vote_label = labels[sort_dist[i],0]  # 取出前k个最小距离对应的label (A or B)
        classcount[vote_label] = classcount.get(vote_label, 0) + 1
        # 字典方法get基于键值获得value，累加+1投票 
    
    # 对classcount字典排序： sorted函数不改变源数据
    sorted_classcount = sorted(classcount.items(), key=operator.itemgetter(1),reverse=True)
    return sorted_classcount[0][0]  # 返回分类结果

# 数据归一化
def autonorm(data):  # 传入一个array而不是dataframe
    minValue = data.min(0)  # 求每列最小值而不是全局最小，因为是每列单独进行归一化
    maxValue = data.max(0)  # 求每列最大值，得到一个一维array
    ranges = maxValue - minValue  # 求极差(3,)
    norm_zero = np.zeros(data.shape)  # 生成一个跟传入array一样大的全0数组
    m = data.shape[0]           # 最后求归一化数据 (data-min)/(max-min)
    norm_data = (data - np.tile(minValue, (m,1)))/np.tile(ranges, (m,1)) 
    return norm_data, ranges, minValue

# 距离加权方法1: 高斯加权
def gaussian(dist, a=1, b=0, c=0.3):
    return a * math.e ** (-(dist - b) ** 2 / (2 * c ** 2))

