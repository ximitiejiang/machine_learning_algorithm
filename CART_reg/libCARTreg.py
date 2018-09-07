#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:21:05 2018

@author: suliang

一些教训：
源程序使用了很多mat格式进行运算，我的理解是mat格式做矩阵运算跟matlab相近，点积写法简单不用写dot
但mat引入后我的程序中大量的set()出错，因为mat嵌套后都不是iterable.
而核心原因就在于array和mat两种格式在切片以后输出不一样
arr[:,-1]输出的是一维array
mat[:,-1]输出的是二维matrix, 而这个二维matrix是unhashable的
源程序也碰到这类问题了，源办法是mat.T.tolist()[0]，如果是array同样操作后结果却不一样
所以用mat就不要跟array混用，不然在这点上会把自己搞混乱。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# 此处采用取平均的方式生成叶子结点
def regLeaf(data):  # 生成回归叶子结点: 此处是做回归模型，所以就取的是该子集的平均值
    return np.mean(data[:,-1])

# 此处采用均方误差计算切分后每个子集的均方误差
def regErr(data):  # 生成均方误差
    return np.var(data[:,-1]) * data.shape[0]

# 此处采用线性回归模型来拟合叶子结点的所有点
def linearSolve(data):
    m, n = data.shape
    X = np.mat(np.ones((m,n)))
    y = np.mat(np.ones((m,1)))
    X[:,1:n] = data[:,0:n-1]  # 第一列已经取1，说明对应的是截距，theta=[theta0, theta1]
    y = data[:,-1].reshape(-1,1)
    
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        print('matrix can not inverse')
        return
    theta = xTx.I * (X.T * y)
    return theta, X,y    
# 此处采用线性回归模型来生成叶子结点
def modelLeaf(data):
    theta, *rest = linearSolve(data) 
    return theta
# 此处采用方差之和作为误差评价
def modelErr(data):
    theta, X, y = linearSolve(data)
    y_test = X*theta
    error = sum(np.power(y - y_test, 2))
    return error    


def chooseBestSplit(data, leafType=regLeaf, errType = regErr, ops = (1,4)):
    tolS = ops[0]   # 容许的误差下降值阀值为1
    tolN = ops[1]   # 切分的最少样本数阀值为4
    data = np.mat(data)
    if len(set(data[:,-1].T.tolist()[0])) == 1:  
    # 这句语法较复杂，最后一列转秩转list为一个list嵌套list，set不认嵌套，所以取第0个元素
        return None, leafType(data) # 如果子数据集分类标签值一样则不再迭代，返回该子数据集的均值作为叶结点
    m,n = data.shape
    S = errType(data)  # 计算均方误差
    bestS = np.inf   # 初始均方误差为无穷大
    bestIndex = 0 
    bestValue = 0
    
    for featIndex in range(n-1):   # 外循环：所有特征向量（取到n-1是因为最后一列是标签）
        for splitVal in set(data[:, featIndex].T.tolist()[0]): # 内循环：在该特征列循环取每一个值 
            # 内循环的语法有点复杂，因为取出该列后是一个嵌套list的matrix，无法set,只能转秩转list取第0个元素才iterable
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

# 只有预测函数singlePointForcase迭代到叶子结点，才会调用如下两个预测值计算函数
# 此为针对普通CART树的评价函数: 输入树和单点数值，直接返回叶子结点值(因为回归树的叶子结点存的就是平均值)
def regTreeEval(model, inDat):
    return float(model)
# 此为针对CART模型树的评价函数: 输入树和单点数值，此时树已经是一个线性模型，则返回线性模型计算的
def modelTreeEval(model, inDat):
    n = inDat.shape[1]
    X = np.mat(np.ones((1, n+1)))  
    X[:,1:n+1] = inDat   # 把数据inDat处理成线性回归模型认可的数据格式，首列为1
    return float(X*model)  # 返回计算出了y = theta * X

# 用CART回归做预测
def singlePointForcast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree):  # 如果不是一棵树，则调用叶子结点预测值计算公式
        return modelEval(tree, inX)    
    # 如果是树，且如果测试数据的第0个值(即x值)大于树结点的切分值，则说明测试数据属于左子树
    # 由于inData是matrix嵌套list，要取到第一个元素值，需要tolist()以后取第0个位置才能取消嵌套再取0元素
    if inData.tolist()[0][tree['spInd']] > tree['spVal']: 
        if isTree(tree['left']): # 如果左子树还是树，继续迭代
            return singlePointForcast(tree['left'], inData, modelEval)
        else:  # 如果左子树是叶子结点，返回
            return modelEval(tree['left'], inData)    
    # 如果是树，且如果测试数据的第0个值(即x值)小于树结点的切分值，则说明测试属于属于右子树
    else:
        if isTree(tree['right']):
            return singlePointForcast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

    
def groupPointsForcast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    y_test = np.mat(np.zeros((m,1)))
    for i in range(m):   # 循环计算每一个测试点的预测值
        y_test[i,0] = singlePointForcast(tree, np.mat(testData[i]), modelEval)
    return y_test

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

def test_modelTree():  # 采用exp2.txt的数据检查CART模型树的效果
    filename = 'exp2.txt'
    data = loadDataSet(filename)
    modelTree = createTree(data, leafType = modelLeaf, errType = modelErr, ops = (1,10))  # 创建一棵误差下降阀值为1，最少样本数为10的树
    return data, modelTree
    

def test_bike():   # 采用自行车与智商数据，进行几种不同树的性能对比，同时完成预测，以及可视化
    filename1 = 'bikeSpeedVsIq_train.txt'
    filename2 = 'bikeSpeedVsIq_test.txt'
    trainMat = np.mat(loadDataSet(filename1))
    testMat = np.mat(loadDataSet(filename2))

    # 创建一棵CART普通回归树
    regTree = createTree(trainMat, leafType = regLeaf, errType = regErr, ops = (1,20))  
    #y_temp = singlePointForcast(regTree, testMat[0], modelEval = regTreeEval)    
    y_test = groupPointsForcast(regTree, testMat[:,0])    
    plt.figure(figsize = (6,4), dpi=80)
    plt.scatter(np.array(trainMat)[:,0],np.array(trainMat)[:,1], c = 'g')
    z = sorted(zip(np.array(testMat)[:,0], np.array(y_test)))  # 排序从小到大
    z = np.array(z)
    plt.plot(z[:,0],z[:,1], c = 'b')

    # 创建一棵CART模型树
    modelTree = createTree(trainMat, leafType = modelLeaf, errType = modelErr, ops = (1,20))
    y_test2 = groupPointsForcast(modelTree, testMat[:,0],modelEval = modelTreeEval) 
    plt.figure(figsize = (6,4), dpi=80)
    plt.scatter(np.array(trainMat)[:,0],np.array(trainMat)[:,1], c = 'g')
    z = sorted(zip(np.array(testMat)[:,0], np.array(y_test2)))  # 排序从小到大
    z = np.array(z)
    plt.plot(z[:,0],z[:,1], c = 'b')
    
    return trainMat, testMat, regTree, modelTree, y_test
    

# ------运行区------------------------------------------------------    
if __name__ == '__main__':
    
    test_id = 4  # 程序运行前，需要指定test_id
    
    if test_id == 0:  # 简单数据集，测试最简单两层CART树的生成
        data, myTree = test_ex00()
    
    elif test_id == 1: # 测试分支相对多的一棵CART树生成
        data, myTree = test_ex0()
    
    elif test_id == 2:  # 测试剪后枝算法
        data, biggestTree, newTree = test_prune()

    elif test_id == 3:  # 测试CART模型树算法
        data, modelTree = test_modelTree()  
    
    elif test_id == 4:  # 在自行车智商实例上对比普通CART回归和CART模型树回归
        trainMat, testMat, regTree, modelTree, y_test = test_bike() 
        
    
    else:
        print('Wrong test_id!')





