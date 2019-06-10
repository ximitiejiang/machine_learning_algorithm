#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:59:17 2018

@author: suliang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# 导入数据
n_dots = 40
X = 5*np.random.rand(n_dots,1)
y = np.cos(X).ravel()
y += 0.2 * np.random.rand(n_dots) - 0.1

# 设置KNN算法参数
k =5
reg = KNeighborsRegressor(k)
reg.fit(X,y)
print(reg.score(X,y))

# 预测
T = np.linspace(0,5,500)[:,np.newaxis]  # 把T格式从行的方式[1,2,3]变为列的方式[[1],[2],[3]]
y_pred = reg.predict(T)

# 可视化
plt.figure(figsize=(8,5),dpi=80)
plt.scatter(X,y,c='g',s=100,cmap='cool', label='data')
plt.plot(T,y_pred,c='k',label = 'prediction', linewidth = 2)


