#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:27:18 2018

@author: suliang
"""

# lib of AdaBoost
import numpy as np

def loadSimpleData():
    data = np.mat([[1.0, 2.1],
                  [2.0, 1.1],
                  [1.3, 1.0],
                  [1.0, 1.0],
                  [2.0, 1.0]])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data, labels


def stumpClassify(data, dimen, threshVal, threshIneq): # stump树桩
    retArray = np.ones((data.shape[0],1))
    if threshIneq == 'lt':
        retArray[data[:,dimen] <= threshVal] = -1.0
    else:
        retArray[data[:,dimen] > threshVal] = -1.0
    return retArray

        
def buildStump(data, labels, D):  # 构建树桩，用于作为基学习器。D代表样本权重向量
    data = np.mat(data)
    labels = np.mat(labels).T
    m,n = data.shape
    
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    
    for i in range(n):     # 外层循环，遍历每列特征
        rangeMin = data[:,i].min()
        rangeMax = data[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        
        for j in range(-1, int(numSteps)+1):  # 中层循环，遍历
            
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMax + float(j) * stepSize)
                predictedVals = stumpClassify(data, i, threshVal, inequal)
                
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labels] = 0
                weightedError = D.T * errArr
                print('split')
                
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst
            
    



def test():
    data, labels = loadSimpleData()
    D = np.mat(np.ones((5,1))*0.2)
    buildStump(data, labels, D)
    
test()