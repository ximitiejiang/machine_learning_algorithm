#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:14:01 2018

@author: suliang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 生成一组数据
n_dots = 200
X = np.linspace(-2*np.pi, 2*np.pi, n_dots)
Y = np.sin(X) + 0.2*np.random.rand(n_dots) - 0.1      
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

# 线性回归模型
# 回归模型评价方式：score函数，mean_squared_error(y_real,y_predict)函数
reg = LinearRegression(normalize = True)
reg.fit(X, Y)
train_score = reg.score(X,Y)
mse = mean_squared_error(Y, reg.predict(X))
print(f'train score is {train_score}; mse is {mse}.')

# 回归可视化
reg.coef_.shape
reg.coef_[0]
plt.figure(figsize=(4,4),dpi=80)
plt.scatter(X,Y)
plt.plot(X, reg.predict(X),c='r')


# 对比多个线性回归的效果---------------------------------------
X_train = []
X_train.append(X)
X_train.append(PolynomialFeatures(degree=1).fit_transform(X))
X_train.append(PolynomialFeatures(degree=2).fit_transform(X))
X_train.append(PolynomialFeatures(degree=4).fit_transform(X))

model = []
model.append(('reg_lir',LinearRegression(normalize = True)))
model.append(('reg_poly_1',LinearRegression(normalize = True)))
model.append(('reg_poly_2',LinearRegression(normalize = True)))
model.append(('reg_poly_4',LinearRegression(normalize = True)))

results = []
for name, m in enumerate(model):
    #kf = KFold(n_splits = 10) 
    #cv_score = cross_val_score(model, X_train, y_train, cv=kf)
    m.fit(X_train,Y)
    results.append((name, m.score, mean_squared_error(Y, m.predict(X))))

# 显示结果
n = len(results)    
for i in range(n):
    print(f'Model name: {results[i][0]}; test score: {result[i][1]}; mse: {result[i][2]}')

# 可视化结果
    