#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:06:54 2018

@author: suliang
"""

import numpy as np
import pandas as pd

import libKNN

    
# 测试算法
group, labels = libKNN.createDataSet()
result = libKNN.clf_point([0,0],group, labels, 3)
print('clf result: {}'.format(result))

# 读入数据
data = pd.read_table('datingTestSet2.txt',header=None)
print(data.head())
# 拆分数据
X = data.loc[:,[0,1,2]].values
y = data.loc[:,[3]].values
# 归一化数据
X_norm, ranges, minValue= libKNN.autonorm(X)    # 归一化，传入一个array
# 基于clf_point模型进行数据集计算
test_Ratio = 0.2   # 定义测试集的百分比：一般20%做测试集，80%做训练集
m = X_norm.shape[0]
num_test = int(m*test_Ratio)  # 测试集的个数:前20%用来测试
error_count = 0.0  # 错误个数统计

# 计算分类结果
for i in range(num_test):  # 循环取出每一个测试样本
    # 调用KNN算法训练：后80%用来训练
    # 每个数据跟这80%训练数据都进行距离计算，挑出前k个进行投票
    clf_result = libKNN.clf_point(X_norm[i,:], X_norm[num_test:m,:], y[num_test:m],3)
    print('clf rusult is: {}; the real result is: {}'.format(clf_result, y[i]))
    # 计算错误总数
    if (clf_result != y[i]):
        error_count += 1.0
print('the total error rate is: {}'.format(error_count/float(m)))  # 统计错误率

'''
plt.figure(figsize=(6,5), dpi = 80)
plt.scatter(X[:,1], X[:,2], 15.0*np.array(y).reshape(1,-1))
plt.legend(loc="best")
plt.show()
'''
    
    
