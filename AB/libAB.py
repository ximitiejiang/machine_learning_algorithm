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
    retArray = np.ones((data.shape[0],1))  # 先都设为1
    if threshIneq == 'lt':
        retArray[data[:,dimen] <= threshVal] = -1.0  # 如果是小于分割点，则
    else:
        retArray[data[:,dimen] > threshVal] = -1.0   # 大于阀值的归到
    return retArray

        
def buildStump(data, labels, D):  # 构建树桩，用于作为基学习器。D代表样本权重向量
    data = np.mat(data)
    labels = np.mat(labels).T
    m,n = data.shape
    
    numSteps = 10.0    # 特征所有可能值上进行遍历
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    
    for i in range(n):     # 外层循环，遍历每列特征
        rangeMin = data[:,i].min()
        rangeMax = data[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        
        for j in range(-1, int(numSteps)+1):  # 中层循环，遍历该特征最小最大值之间
                                              # 找到最合适的分割点
            for inequal in ['lt', 'gt']:   # 内层循环，用于切换不等式<=, >=
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(data, i, threshVal, inequal)
                # stumpClassify在指定特征的指定分割点，分别计算大于和小于分割点
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labels] = 0  #预测正确的取0
                weightedError = D.T * errArr
                print('the weightedErr = {}'.format(weightedError))
                
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst
            

def adaBoostTrainDS(data, labels, numIter = 40):  # 输入数据，标签，迭代次数
    weakClassArr = []  # 
    m = data.shape[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIter):
        bestStump, error, classEst = buildStump(data, labels, D)
        print('D:')
        alpha = float(0.5*log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst')
        
        expon = multiply(-1 * alpha*mat(labels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        
        aggClassEst += alpha*classEst
        print('aggClassEst')
        
        aggErrors = multiply(sign(aggClassEst)) != mat(labels).T, np.ones((m,1))
        errorRate = aggErrors.sum()/m
        print('total error')
        if errorRate == 0.0:
            break
    return weakClassArr

        
        
        
        



def test():
    data, labels = loadSimpleData()
    D = np.mat(np.ones((5,1))*0.2)
    buildStump(data, labels, D)
    
test()