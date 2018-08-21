#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:24:47 2018

@author: suliang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

import libKNN

centers = np.array([[-2,2],[2,2],[0,4]])
X,y = make_blobs(n_samples=120, centers=centers,random_state=0, cluster_std=0.60)

plt.figure(figsize=(8,5),dpi=80)
plt.scatter(X[:,0],X[:,1],c='y',s=100,cmap='cool')
plt.scatter(centers[:,0],centers[:,1],c='r',s=100,marker='^')
#plt.scatter(X_sample[0][0],X_sample[0][1],c='r',s=100,marker='x')

# 数据量较少，就不划分数据集了

# 进行单点预测
inX = [0,3]

for i in range(X.shape(0)):
    libKNN.clf_point(inX, X, y, 5)
    