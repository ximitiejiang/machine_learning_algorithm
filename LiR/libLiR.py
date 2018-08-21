#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 09:04:41 2018

@author: suliang

libLiR线性回归库文件解释(参考了github上面lawlite19的源码)
    - clf(data, alpha=0.01, num_iters=400), 线性回归分类器核心算法
    - BGD(X,y,theta,alpha,num_iters), 批量梯度下降算法BGD
    - 
"""

import numpy as npa
import pandas as pd


# 线性回归算法，输入data为array数组，alpha为梯度学习的学习率，num_iters为梯度学习的最大迭代次数
def clf(data, alpha=0.01, num_iters=400):
    
    X = data[:,0:-1]      # X对应0到倒数第2列                  
    y = data[:,-1]        # y对应最后一列  
    m = len(y)            # 总的数据条数
    col = data.shape[1]      # data的列数
    
    # 考虑把归一化移出LiR算法，数据可以在外边归一化
    #X,mu,sigma = featureNormaliza(X)    # 归一    
    X = np.hstack((np.ones((m,1)),X))    # 在X前加一列1    
    theta = np.zeros((col,1))
    y = y.reshape(-1,1)   #将行向量转化为列（这里注释不对，是把一维array转化成二维array）
    
    theta,J_history = BGD(X, y, theta, alpha, num_iters)
    plotJ(J_history, num_iters)
    
    return mu,sigma,theta   #返回均值mu,标准差sigma,和学习的结果theta

# BGD批量梯度下降算法
# 输入: X为训练数据，y为labels, alpha为学习率，num_iters为最大迭代次数
def BGD(X,y,theta,alpha,num_iters):
    m = len(y)      
    n = len(theta)    
    temp = np.matrix(np.zeros((n,num_iters)))   # 暂存每次迭代计算的theta，转化为矩阵形式    
    J_history = np.zeros((num_iters,1)) #记录每次迭代计算的代价值    
    for i in range(num_iters):  # 遍历迭代次数    
        h = np.dot(X,theta)     # 计算内积，matrix可以直接乘
        temp[:,i] = theta - ((alpha/m)*(np.dot(np.transpose(X),h-y)))  #梯度的计算
        theta = temp[:,i]
        J_history[i] = computerCost(X,y,theta)      #调用计算代价函数
        print('.',)      
    return theta,J_history 

def computerCost(X,y,theta):
    m = len(y)
    J = 0
    
    J = (np.transpose(X*theta-y))*(X*theta-y)/(2*m) #计算代价J
    return J


# SGD随机梯度下降法
def SGD():
    pass


# MBGD小批量梯度下降法
def MBGD():
    pass


def autonorm(data):  # 传入一个array而不是dataframe
    minValue = data.min(0)  # 求每列最小值而不是全局最小，因为是每列单独进行归一化
    maxValue = data.max(0)  # 求每列最大值，得到一个一维array
    ranges = maxValue - minValue  # 求极差
    norm_zero = np.zeros(data.shape)  # 生成一个跟传入array一样大的全0数组
    m = data.shape[0]           # 最后求归一化数据 (data-min)/(max-min)
    norm_data = (data - np.tile(minValue, (m,1)))/np.tile(ranges, (m,1)) 
    return norm_data, ranges, minValue

# 画每次迭代代价的变化图
def plotJ(J_history,num_iters):
    x = np.arange(1,num_iters+1)
    plt.plot(x,J_history)
    plt.xlabel(u"迭代次数",fontproperties=font) # 注意指定字体，要不然出现乱码问题
    plt.ylabel(u"代价值",fontproperties=font)
    plt.title(u"代价随迭代次数的变化",fontproperties=font)
    plt.show()

# 测试linearRegression函数
def testLinearRegression():
    mu,sigma,theta = linearRegression(0.01,400)
    #print u"\n计算的theta值为：\n",theta
    #print u"\n预测结果为：%f"%predict(mu, sigma, theta)
    
# 测试学习效果（预测）
def predict(mu,sigma,theta):
    result = 0
    # 注意归一化
    predict = np.array([1650,3])
    norm_predict = (predict-mu)/sigma
    final_predict = np.hstack((np.ones((1)),norm_predict))
    
    result = np.dot(final_predict,theta)    # 预测结果
    return result

# 测试主程序
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读入数据
data0 = pd.read_table('ex0.txt',header = None)
data = data0.values[:,1:3]

X0 = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)
plt.figure(figsize=(6,4),dpi=100)
plt.scatter(X0,y,c='g',s=100,cmap='cool')

# 归一化数据
#X = autonorm(X0)
#plt.scatter(X,y)

# 基于算法训练模型
theta, J_history = clf(X0, alpha=0.01, num_iters=400)

a = np.array([1,2,3])
b = np.array([4,5,6])
c1 = np.stack((a,b), axis =0)
c2 = np.hstack((a,b))
c3 = np.vstack((a,b))
