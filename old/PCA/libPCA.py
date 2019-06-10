#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 21:39:34 2018

@author: suliang


PCA主要用来给特征降维
PCA的主成分选择原则是：主成分代表了方差最大的方向; 各个主成分之间相互正交

Q.做PCA之前，为什么要去均值：
    * 
    
Q.PCA的求解理论应该是算出特征值和特征向量，找到特征值最大的前n个对应特征向量，乘以源数据得到降维数据。
为什么在PCA实际求解释，却都是采用求协方差矩阵的SVD方法？
    *

Q.PCA的推导：
    * 
    
"""
from numpy import *

def loadDataset():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=10000, 
                      n_features=3, 
                      centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], 
                      cluster_std=[0.2, 0.1, 0.2, 0.2],
                      random_state =9)
    return X, y


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat


if __name__ == '__main__':
    
    test_id = 0
    
    if test_id == 0:
        data, label = loadDataset()
        
        plt.scatter(data[:,0],data[:,1])
        
    elif test_id == 1:  # 对方差进行排序，查看前n个占比最高方差情况跟PCA结果关系
        pass
    
    else:
        print('Wrong test_id!')
        
        