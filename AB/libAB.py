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
        retArray[data[:,dimen] <= threshVal] = -1.0  # 如果是小于模式，则小于分割点为-1
    else:
        retArray[data[:,dimen] > threshVal] = -1.0   # 如果是大于模式，则大于分割点为-1
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
                                              # 找到最合适的分割点，取-1开始是分割点
            for inequal in ['lt', 'gt']:   # 内层循环，用于切换不等式<=, >=
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(data, i, threshVal, inequal)
                # stumpClassify在指定特征的指定分割点，分别计算大于和小于分割点
                # stumpClassify返回的就是模型分类结果bestClasEst(跟实际label会有误差)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labels] = 0  #预测正确的取0
                weightedError = D.T * errArr  # 获得error的叉乘累加
                
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy() # 最优模型的分类结果
                    bestStump['dim'] = i               # 最优模型的特征位置
                    bestStump['thresh'] = threshVal    # 最优模型的切分点
                    bestStump['ineq'] = inequal        # 最优模型的不等式类型
                    bestStump['minError'] = minError   #
                    
    return bestStump, minError, bestClasEst  
    # 最终通过遍历所有特征(外循环)，遍历每个特征的值范围(中循环)，遍历大于和小于的情况
    # 最终得到一个弱分类器（一个只有一个特征的一个分割点的树桩）
            

def adaBoostTrainDS(data, labels, numIter = 40):  # 输入数据，标签，迭代次数
    weakClassArr = []  # 
    m = data.shape[0]
    D = np.mat(np.ones((m,1))/m)  # 先定义每个样本的初始权重（让D的和为1）
    aggClassEst = np.mat(np.zeros((m,1)))
    
    for i in range(numIter):
        bestStump, error, clasEst = buildStump(data, labels, D)
        alpha = float(0.5*np.log((1.0 - error)/max(error, 1e-16))) #为了防止err=0，分母处理了以下
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('beststump = {}'.format(bestStump))
        expon = np.multiply(-1 * alpha*np.mat(labels).T, clasEst) # 计算指数
        # expon = 如果是正确样本则取-alpha, 如果是错误样本则取alpha
        D = np.multiply(D, np.exp(expon))  # 更新样本权重D
        D = D/D.sum()                # 更新样本权重D
        print('D = {}'.format(D))
        
        aggClassEst += alpha * clasEst  # 此向量计算多个弱分类器输出的线性加权之和      
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(labels).T, np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print('total error rate = {}'.format(errorRate))
        print('_'*30)
        if errorRate == 0.0:
            break
    return weakClassArr   
    # 每一次循环Iter，生成一个bestStump
    # 所以weakClass里边弱分类器个数跟循环次数相同，除非错误率=0提前跳出循环


def adaClassify(inX, classifierArr):  # 输入待分类数据，和训练好的弱分类器
    inX = np.mat(inX)
    m = inX.shape[0]
    aggClassEst = np.mat(np.zeros((m,1)))  # 这个变量用来累加各分类器输出的加权之和
    
    for i in range(len(classifierArr)):  # 循环使用多个弱分类器
        classEst = stumpClassify(inX, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])
        # 用某个弱分类器获得一个分类输出(输入和输出都是对多个样本同时分类)
        aggClassEst += classifierArr[i]['alpha']*classEst
        # 把每个弱分类器的预测输出进行线性加权累加
        print(aggClassEst)
    return np.sign(aggClassEst)  # 把线性加权累加值的符号作为最终预测结果
                  

def test():
    data, labels = loadSimpleData()
    D = np.mat(np.ones((5,1))*0.2)
    
    weakClassArr = adaBoostTrainDS(data, labels, numIter = 10)
    print('total classifiers: {}'.format(weakClassArr))
    return weakClassArr
    


    
weakClassArr = test()
adaClassify([0,0], weakClassArr)    
    
    