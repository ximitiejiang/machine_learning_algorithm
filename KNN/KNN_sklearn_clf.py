#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:59:17 2018

@author: suliang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# 导入数据
centers = np.array([[-2,2],[2,2],[0,4]])
X,y = make_blobs(n_samples=120, centers=centers,random_state=0, cluster_std=0.60)

# 设置KNN算法参数
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X,y)
print('classifier score is: {}'.format(clf.score(X,y)))

# 获取近邻点：获得的是近邻点的横坐标坐标标签
X_sample = [[0,2]]  # 要求传入一个2维数据，2维list或者array都可以
y_sample = clf.predict(X_sample)  # 预测
neighbors = clf.kneighbors(X_sample, return_distance=False) # 获得k个近邻点

# 可视化
plt.figure(figsize=(8,5),dpi=80)
plt.scatter(X[:,0],X[:,1],c='y',s=100,cmap='cool')  # 画样本点
plt.scatter(centers[:,0],centers[:,1],c='r',s=100,marker='^')   # 画中心点
plt.scatter(X_sample[0][0],X_sample[0][1],c='r',s=100,marker='x') # 画新样本点
# 画新样本X_sample跟k个近邻点的连线
for i in neighbors[0]: # 绘制每个样本点与近邻的连线
    plt.plot([X[i][0],X_sample[0][0]], [X[i][1], X_sample[0][1]], 'k--', linewidth = 0.6)

# KNN距离加权算法
clf_dis = KNeighborsClassifier(n_neighbors=k, weights = 'distance')
# KNN半径取代距离最近
clf_rad = KNeighborsClassifier(n_neighbors=k, radius = 200.0)  

