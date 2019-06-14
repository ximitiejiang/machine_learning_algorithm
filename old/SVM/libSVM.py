#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:19:27 2018

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


def selectJrand(i,m):
    j = i
    while(j==i):
        j = int(np.random.uniform(0,m))
    return j

# 线性核函数
def linear_kernel(X1, X2):
    res = X1.dot(X2.T)
    return res
 
# 多项式核函数
def poly_kernel(X1, X2):
    res = (self.seta + self.gamma*X1.dot(X2.T))**self.degree
    return res

# 高斯核函数    
def rbf_kernel(X1, X2):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    res = np.exp(-self.gamma * dist.cdist(X1, X2)**2)
    return res


def SMOsimple(data, labels, C, toler, maxIter):
    data = np.mat(data)                      # (100,2)
    labels = np.mat(labels).transpose()      # (100,1)
    m = data.shape[0]
    alphas = np.mat(np.zeros((m,1)))         # (100,1)
    b = 0
    
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, labels).T * \
                  (data * data[i,:].T)) + b
            Ei = fxi - labels[i]
            # 判断所选alphaI是否为支持向量：alphaI>0, alphaI<C，则为支持向量
            # 判断alphaI对应的fxi的误差是否超过所定义偏差，如果超过说明需要优化alpha值
            if ((labels[i]*Ei < -toler) and (alphas[i]< C)) \
               or ((labels[i]*Ei > toler) and (alphas[i]>0)): 
                
                # optimize step1: define alphaIold, alphaJold
                j = selectJrand(i,m)
                fxj = float(np.multiply(alphas, labels).T * \
                      (data * data[j,:].T)) + b
                Ej = fxj - labels[j]
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                
                # optimize step2: calculate L, H, eta
                if (labels[i]==labels[j]):
                    L = max(0, alphas[j] +alphas[i] - C)
                    H = min(C, alphas[j] +alphas[i])
                else:
                    L = max(0, alphas[j] -alphas[i])
                    H = min(C, C +alphas[j] -alphas[i])
                if L==H:
                    continue
                eta = 2.0*data[i,:]*data[j,:].T - data[i,:]*data[i,:].T - \
                      data[j,:]*data[j,:].T
                if eta >=0:
                    continue
                
                # optimize step3: update alphaInew, alphaJnew
                alphas[j] -= labels[j]*(Ei - Ej)/eta 
                if alphas[j] > H:
                    alphas[j] = H
                elif alphas[j]< L:
                    alphas[j] = L
                if ((alphas[j] - alphaJold)<0.00001):
                    continue
                alphas[i] += labels[i]*labels[j]*(alphaJold-alphas[j])
                
                # optimize step4: update b
                b1 = b - Ei - labels[i]*(alphas[i]-alphaIold)* \
                     data[i,:]*data[i,:].T - labels[j]*(alphas[j]-alphaJold)*\
                     data[i,:]*data[j,:].T
                b2 = b - Ej - labels[i]*(alphas[i]-alphaIold)* \
                     data[i,:]*data[j,:].T - labels[j]*(alphas[j]-alphaJold)*\
                     data[j,:]*data[j,:].T
                if alphas[i] > 0 and alphas[i] < C:
                    b = b1
                elif alphas[j] > 0 and alphas[j] < C:
                    b = b2
                else:
                    b = 0.5*(b1 + b2)
                    
                alphaPairsChanged +=1
                
        if (alphaPairsChanged ==0): # 如果不再有alpha进行优化，就最多运行MaxIter次
            iter += 1
        else:
            iter = 0                # 如果有alpha优化过，则重新计算循环次数

    return alphas, b

def SMO(data, labels, C, toler, maxIter):
    pass

def WandPredict(x, data, labels, alphas, b):
    # 计算w
    # 预测
    pass

# -------main-----------
data, labels = loadDataSet('testSet.txt')  # data(100,2), labels(100,)

alphas, b = SMOsimple(data, labels, 100, 0.001, 40) # 

svlist = []
for i in range(len(alphas)):
    if alphas[i]>0:
        svlist.append(i)

# 可视化支持向量
colour = [c*30+60 for c in labels]
plt.figure(figsize = (6,4), dpi=80)
plt.scatter(data[:,0],data[:,1],c=colour)
for i in range(len(svlist)):
    plt.scatter(data[svlist[i],0],data[svlist[i],1], c='r',s=5,linewidths=20,marker='o')        

# 预测+可视化超平面
    
    
