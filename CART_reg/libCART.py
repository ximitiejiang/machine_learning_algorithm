#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:21:05 2018

@author: suliang
"""

import numpy as np
import pandas as pd

def loadDataSet(filename):
    data = pd.read_table(filename,header = None)
    #x = df.iloc[:,:-1].values
    #y = df.iloc[:,-1].values
    data = np.array(data)    
    return data


def binSplitDataSet(data, feature, value):  #
    # 通过np.nonzero()获得该列特征中大于value的值在第几行，取出data[行号, :]
    mat0 = data[np.nonzero(data[:,feature] > value)[0], :]  # 大于value的行
    mat1 = data[np.nonzero(data[:,feature] <= value)[0], :] # 小于等于value的行
    return mat0, mat1


def regLeaf(data):  # 生成回归叶子结点: 此处是做回归模型，所以就取的是该子集的平均值
    return np.mean(data[:,-1])


def regErr(data):  # 生成均方误差
    return np.var(data[:,-1]) * data.shape[0]


def chooseBestSplit(data, leafType=regLeaf, errType = regErr, ops = (1,4)):
    tolS = ops[0]   # 容许的误差下降值阀值为1
    tolN = ops[1]   # 切分的最少样本数阀值为4
    if len(set(data[:,-1].T.tolist())) == 1:  # 如果最后分类标签值一样，则退出
        return None, leafType(data)  # 返回none, 函数值regLeaf(data)
    m,n = data.shape
    S = errType(data)  # 计算均方误差
    bestS = np.inf   # 初始均方误差为无穷大
    bestIndex = 0 
    bestValue = 0
    
    for featIndex in range(n-1):   # 外循环：所有特征向量（取到n-1是因为最后一列是标签）
        for splitVal in set(data[:, featIndex]): # 内循环：在该特征列循环取每一个值 
            
            mat0, mat1 = binSplitDataSet(data, featIndex, splitVal)  #划分两个子数据集
            if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN): #如果数据集内样本个数少于设定值，退出
                continue
            
            # 由于是CART回归，即连续特征，此时用二分法没有问题，但无法用gini指数计算连续特征
            # 因此采用各个子集的均方误差之和来评价和挑选最优特征和最优分割点
            newS = errType(mat0) + errType(mat1)  # 计算两个子数据集的均方误差之和
            if newS < bestS:     # 如果均方误差之和小于之前最优的，则更新最优解
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    
    if (S - bestS) < tolS:   # 如果均方误差的变化小于设定值，说明提升不大了，返回叶子结点
        return None, leafType(data)
    
    mat0, mat1 = binSplitDataSet(data, bestIndex, bestValue) #基于循环得到的最优特征和切分点进行切分
    
    if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):  # 如果切分出的数据集很小，退出
        return None, leafType(data)
    
    return bestIndex, bestValue
    

# 创建一棵树
def createTree(data, leafType = regLeaf, errType = regErr, ops = (1,4)):
    feat, val = chooseBestSplit(data, leafType, errType, ops)  # 选择最优特征和切分值
    if feat == None:      # 如果所有特征都已用完，则返回切分值
        return val
    
    retTree = {}       # 初始树
    retTree['spInd'] = feat  # 在树中存入最优特征
    retTree['spVal'] = val   # 在树中存入切分值
    lSet,rSet = binSplitDataSet(data, feat, val)  # 切分数据集为左右两个数据集
    
    retTree['left'] = createTree(lSet, leafType, errType, ops)  # 存入左子树
    retTree['right'] = createTree(rSet, leafType, errType, ops) # 存入右子树
    return retTree  # 返回树 {{'spInd':xx}, {'spVal':xx}, {'left': 左子树}, {‘right’: 右子树} }


# 树剪枝
def isTree(obj):  # 检测输入变量是否为一棵树，如果是就返回True‘如果不是一棵树，就是叶结点
    return (type(obj).__name__=='dict')

def getMean(tree):  # 递归函数：遍历找到两个叶结点，进行塌陷处理，反馈两个叶结点的平均值
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    
    return (tree['left'] + tree['right'])/2.0

# 剪枝需要基于测试数据，用于评估剪枝后是否能提升模型泛化能力
def prune(tree, testData):  # 剪枝函数：输入待剪枝的树，和测试数据
    if testData.shape[0] == 0:  # 如果没有测试数据，则合并子树
        return getMean(tree)
    
    # 如果左右有一个是子树，则用该特征和分割值在验证数据上划分子集
    if (isTree(tree['right']) or isTree(tree['left'])): 
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    
    # 如果左树是子树，对左树递归进行剪枝(注意，剪枝使用的是左测试子集)
    # 如果右树是子树，对右树递归进行剪枝(注意，剪枝使用的是右测试子集)
    if isTree(tree['left']): 
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): 
        tree['right'] = prune(tree['right'], rSet)
    
    # 如果左右都不是子树，说明左右都是叶子结点，则用该结点特征在验证数据上划分子集
    # 然后计算合并前后的误差（因为是回归，关注的误差是测试集每个值跟叶子点的误差）
    # 合并前计算： 左边叶子值跟测试集误差平方和 + 右边叶子值跟测试集误差平方和
    # 合并后计算：合并值跟测试集误差平方和
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:,-1]-tree['left'], 2)) + \
                       sum(np.power(rSet[:,-1]-tree['right'], 2))
        
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(np.power(testData[:,-1]-treeMean, 2))  
        
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else: return tree
    else: return tree


def regTreeEval():
    return

# 用CART回归做预测
def singlePointForcast(tree, inX, modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inX)
    
    
def forcast(tree, testData, modelEval = regTreeEval):
    pass

# -----------test()-----------
def test_ex00():    # 简单看看数的结构
    filename = 'ex00.txt'
    data = loadDataSet(filename)
    myTree = createTree(data)
    return data, myTree

def test_ex0():   # 样本数较多分支较多的一棵树，CART树能够自动挑选最优特征，无关特征(比如第一列)会忽略
    filename = 'ex0.txt'
    data = loadDataSet(filename)
    myTree = createTree(data)
    return data, myTree

def test_prune():  # 采用ex2训练样本，和ex2test验证样本，因为这个样本的x1,x2量纲不一致，容易产生过拟合
    # 先生成一棵最大树
    filename = 'ex2.txt'
    data = loadDataSet(filename)
    biggestTree = createTree(data, ops=(0,1))  # 创建一棵误差下降阀值为0，最少样本数为1的最大一棵树
    
    filename = 'ex2test.txt'
    testData = loadDataSet(filename)
    # 显示训练样本和测试样本
    import matplotlib.pyplot as plt
    plt.figure(figsize = (6,4), dpi=80)
    plt.scatter(data[:,0],data[:,1], c = 'g')
    plt.scatter(testData[:,0],testData[:,1], c = 'b')
    
    # 开始剪枝
    newTree = prune(biggestTree, testData)
    
    return data, biggestTree, newTree


def test_bike():
    pass

# ---------运行区-------------
data, biggestTree, newTree = test_prune()




