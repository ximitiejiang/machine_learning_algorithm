#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 20:17:01 2018

@author: suliang

libPP = lib of preprocessing
数据预处理
"""

# 数据标准化：变为均值为0，方差为1的高斯标准正态分布
# 数据标准化目的：
# 数据标准化方法：
def dataStandard(X_train):   #
    from sklearn import preprocessing
    import numpy as np
    
    X_scaled = preprocessing.scale(X_train)
    
    X_scaled.mean(axis=0)
    X_scaled.std(axis=0)
    
    return X_scaled
    
# 数据归一化
def dataNormalize():
    pass


# 数据二值化
def dataBinarizer():
    pass


# 特征扩展成多项式特征
def d():
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    poly.fit_transform(X)
    
    
#----运行区--------
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = dataStandard(X_train)