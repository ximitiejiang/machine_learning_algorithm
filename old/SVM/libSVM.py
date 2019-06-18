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
    """随机选择alpha_j"""
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
    """SMO序列最小优化方法：主要出发点就是从alpha的列表中寻找不满足KKT条件的alpha对
    (只要第一个alpha不满足KKT就随机选择alpha对中的第二个alpha)，每循环更新一对alpha对。
    如此循环，直到所有alpha对都满足KKT条件或达到最大更新循环次数，则得到所有alpha的解
    Args:
        data
        label
        C: 代表alpha的最大值限制
        toler: 替代0的一个精度值，判断alpha>0就变成alpha>toler，有助于算法很快收敛
        maxIter: alpha对的更新次数
    """
    data = np.mat(data)                      # (100,2)
    labels = np.mat(labels).transpose()      # (100,1)
    m = data.shape[0]
    alphas = np.mat(np.zeros((m,1)))         # 初始化alpha向量(n_sample, 1) (100,1)
    b = 0
    
    iter = 0
    while (iter < maxIter):  # 外循环： 在指定的maxiter循环次数中(该循环次数是指更新alpha的次数，如果不更新则不算一次)
        alphaPairsChanged = 0  # 是否更新的标志
        for i in range(m):   # 内循环：遍历所有样本的alpha用来作为alpha_i的值
            
            # 1. 遍历寻找alpha_i: 判断alpha_i是否满足KKT条件，如果满足则不需要调整继续选择下一个alpha_i, 如果不满足则进行下面的优化更新
            gxi = float(np.multiply(alphas, labels).T * (data * data[i,:].T)) + b
            Ei = gxi - labels[i]            
            if ((labels[i]*Ei < -toler) and (alphas[i]< C)) \
               or ((labels[i]*Ei > toler) and (alphas[i]>0)):  # 不符合KKT条件的3种情况(yEi<0&alpha<C, yEi>0&alpha>0, yEi=0&alpha=0orC)
                                                               # 这里引入toler是为了让KKT达到一定精度toler就停止，如果要完全等于0，往往不能很快收敛
            # 2. 随机寻找alpha_j
                j = selectJrand(i,m)
                fxj = float(np.multiply(alphas, labels).T * \
                      (data * data[j,:].T)) + b
                Ej = fxj - labels[j]
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                
            # 3. 基于alpha_i/alpha_j计算L, H, eta
                if (labels[i]==labels[j]):
                    L = max(0, alphas[j] +alphas[i] - C)
                    H = min(C, alphas[j] +alphas[i])
                else:
                    L = max(0, alphas[j] -alphas[i])
                    H = min(C, C +alphas[j] -alphas[i])
                if L==H:
                    continue
                eta = 2.0*data[i,:]*data[j,:].T - data[i,:]*data[i,:].T - \
                      data[j,:]*data[j,:].T  # eta即 eta = K11+K22-2K12, 用于更新alpha_j
                if eta >=0:
                    continue
                
            # 4. 基于eta更新alpha_i, alpha_j
                alphas[j] -= labels[j]*(Ei - Ej)/eta 
                if alphas[j] > H:
                    alphas[j] = H
                elif alphas[j]< L:
                    alphas[j] = L
                if ((alphas[j] - alphaJold)<0.00001):
                    continue
                alphas[i] += labels[i]*labels[j]*(alphaJold-alphas[j])
                
            # 5. 基于更新的alpha_i,alpha_j更新b
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

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros(self.m, 1))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
    
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alpha, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)
        
        

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
    
    
