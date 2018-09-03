#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 20:17:01 2018

@author: suliang

libPP = lib of preprocessing
数据预处理
"""

'''
标准化：也叫z-score, 变为均值为0，方差为1的高斯标准正态分布
标准化方法：xhat = (x - mu)/theta (减均值，除方差)
标准化特点：改变不同特征维度到
'''
def dataStandard(X):   #
    from sklearn import preprocessing
    import numpy as np
    
    X = np.array([[ 1., -1.,  200.],
                 [ 2.,  0.,  100.],
                 [ 0.,  1., -100.]])
    X_scaled = preprocessing.scale(X_train)

    X_scaled.mean(axis=0)
    X_scaled.std(axis=0)
    
    return X_scaled
    

'''
归一化：也叫min-max normalization
归一化方法：xhat = (x - min)/(max - min)
归一化特点：改变不同特征的维度到(0,1)之间，优点是提高迭代速度和精度，缺点是容易受异常点影响，鲁棒性差
'''
def dataNormalize():
    from sklearn import preprocessing
    import numpy as np
    X = np.array([[ 1., -1.,  200.],
                 [ 2.,  0.,  100.],
                 [ 0.,  1., -100.]])
    X_normalized = preprocessing.normalize(X, norm='l2')
    
    return X_normalized


# 数据二值化
def dataBinarizer():
    pass


# 特征扩展成多项式特征
def d():
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    poly.fit_transform(X)
    
    
#----运行区--------

X_scaled = dataStandard(X_train)