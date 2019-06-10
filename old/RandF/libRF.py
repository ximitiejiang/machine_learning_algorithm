#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:45:23 2018

@author: suliang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 载入数据
def classifyData_1():
    from sklearn.datasets.samples_generator import make_classification
    X,labels=make_classification(n_samples=200,n_features=2,n_redundant=0,
                                 n_informative=2, random_state=1,
                                 n_clusters_per_class=2)
    rng=np.random.RandomState(1)
    X += rng.uniform(size=X.shape)
    
    return X, labels


def loadDataSet1(filename):  # 用来加载<统计学习方法>中的贷款申请数据集
    fr = open(filename)
    data = [inst.strip().split('\t') for inst in fr.readlines()]
    featName = ['age', 'job', 'house','credit']
    return data, featName


# 从所有样本中有放回选出m x k的样本子集：m行，k列
def choose_samples(data, k):
    import random as rd
    import math
    
    m,n = data.shape
        
    feature = []
    for j in range(k):
        feature.append(rd.randint(0, n-2))  # 随机选出k个特征的index
        
    index = []
    for i in range(m):
        index.append(rd.randint(0, m-1))  # 随机选出m个样本的index
    
    data_samples = []
    for i in range(m):  # 循环m个样本
        data_temp = []
        for feat in feature: # 循环n个特征
            data_temp.append(data[index[i]][feat]) #取到一个样本并放入data_temp
            
        data_temp.append(data[index[i]][-1])   # 取到
        data_samples.append(data_temp)
    return data_samples, feature  # 返回data_samples为list嵌套


def majorityCount(classList):  # 采用一种更简洁优雅的写法来获得最多label的出现次数
    from collections import Counter
    num_count = Counter(classList)
    max_count = max(zip(num_count.values(), num_count.keys()))[0]
    return max_count


def calcGini(data):  # 计算一个数据集的基尼指数，计算基尼指数只跟数据集的标签列相关，所以只要data传入
    from collections import Counter
    numEntries = len(data)
    labelCounts = Counter(np.array(data)[:,-1])
    
    gini = 1.0
    for key in labelCounts:        
        gini = gini - pow(labelCounts[key]/numEntries,2)
    return gini


def splitDataSet(spdata, axis, value):  
    # 划分数据集，输入数据集data, 特征列号axis，对应特征列的特征值
    # 比如splitDataSet(data,0,1)代表第0列特征值为1的数据子集
    import copy
    subDataSet = []
    restDataSet = copy.deepcopy(spdata)
    deltimes = 0
    for i, row in enumerate(spdata):   # 循环每行
        if row[axis] == value:  # 如果该行对应列等于划分值则取出该行
            reducedFeatVec = row[:axis]    # 取出axis列左的数据(不包括axis列)
            reducedFeatVec.extend(row[axis+1:]) # 取出axis列右的数据(不包括axis列)
            subDataSet.append(reducedFeatVec)
            # 每删除一行，计算行的起点就往回调1
            del restDataSet[i-deltimes]
            deltimes += 1    
    # 去除该特征列
    for i in range(len(restDataSet)):
        del restDataSet[i][axis] # 
    # 这个地方实现掉进无数坑了：如果同步删除某列，结果删除后的行不对了
    # 如果一次性删除首列后做差集，又因为有相同行，差集数据又不对    
    return subDataSet, restDataSet   # 返回指定划分的2个数据集


def chooseBestFeatureToSplit(data):  
    numFeatures = len(data[0]) - 1   # 获得特征个数，label不算所以减一
    baseGini = calcGini(data)
    bestGiniGain = 0.0
    bestFeature = -1
    bestValue = -1
    
    for i in range(numFeatures):  # 外循环定义特征i
        featList = [sample[i] for sample in data] # 取出第i列特征
        uniqueVals = set(featList)  # set函数是获得不重复的值（即去除重复）
        
        newGini = 0.0
        for value in uniqueVals:  # 内循环定义该特征i的取值种类
            subDataSet, restDataSet = splitDataSet(data, i, value) # 得到该特征该值划分的2个子集        
            p1 = len(subDataSet)/float(len(data))
            p2 = len(restDataSet)/float(len(data))
            newGini = p1*calcGini(subDataSet) + p2*calcGini(restDataSet) # 计算划分子数据集的熵
        
            giniGain = baseGini - newGini # 计算特征i划分的基尼增益(用增益判断就不用记下所有数据做排序)
            if (giniGain > bestGiniGain): # 如果特征i的增益最大，则以该特征i为最优特征
                bestGiniGain = giniGain
                bestFeature = i
                bestValue = value
                
    return bestFeature, bestValue


def createTree(data, featName):  # 创建树输入data必须是带label的数据：本质上创建树是把所有数据存储起来了
    featColumnName = featName[:]  
    # 在函数体内修改了形参featName,为了防止对体外同名变量的影响需要复制
    # 由于featName只是一个单层list,采用浅拷贝就够了    
    classList = np.array(data)[:,-1].tolist()  # 取出最后一列的标签值
       
    if classList.count(classList[0])==len(classList):  # 如果所有标签值相同，说明只有唯一分类，可退出循环
        return classList[0]
    if len(data[0])==1:   # 如果是遍历到了最后，data[0]就是
        return majorityCount(classList)  # 就返回次数最多的分类值
    
    bestFeat, bestValue = chooseBestFeatureToSplit(data)
    bestfeatName = featColumnName[bestFeat]
    
    myTree = {bestfeatName:{}}  # 更新树的key
    del(featColumnName[bestFeat])  # 已经split过的特征名称就去掉,就是此处修改了形参
    
    subSet, restSet = splitDataSet(data, bestFeat, bestValue)
    featValues = [example[bestFeat] for example in data] # 获得最佳特征的列
    uniqueValues = set(featValues)  # 去除重复值
    
    for value in uniqueValues:  #
        if value==bestValue:
            myTree[bestfeatName][value]=createTree(subSet,featColumnName)
        else:
            myTree[bestfeatName][value]=createTree(restSet,featColumnName)
        
    return myTree


def predict(sample, tree):
    if tree.results != None:
        return tree.results
    else:
        val_sample = sample[tree.feat]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return predict(sample, branch)        

    
def randomForest_traing(data, num_tree):  # 训练数据， 需要构建的树的数量
    import random as rd
    import math    
    
    tree_result = []
    tree_feature = []
    
    n = data.shape[1]
    if n > 2:
        k = int(math.log(n-1, 2)) + 1
    else:
        k = 1  # 如果只有2个特征，则取其中1个特征构建树
        
    for i in range(num_tree):
        data_samples, feature = choose_samples(data, k)
        tree = build_tree(data)
        tree_result.append(tree)
        
    tree_feature.append(feature)
    
    return tree_result, tree_feature



# -------------------运行区-------------------------------------------
if __name__ == '__main__':
    
    test_id = 2    # 程序运行前，需要指定test_id
    
    if test_id == 0:  # 调试划分数据集子函数
        filename = 'loan.txt'
        data, featName = loadDataSet1(filename)
        sub, rest = splitDataSet(data, 2, 'yes')
    
    elif test_id == 1:  # 调试选择最优特征
        filename = 'loan.txt'
        data, featName = loadDataSet1(filename) # 测试最优特征选择
        bestFeature, bestValue = chooseBestFeatureToSplit(data)
        
    elif test_id == 2: # 整体调试CART_clf
        filename = 'loan.txt'
        data, featName = loadDataSet1(filename) # 测试最优特征选择
        myTree = createTree(data, featName)
        
    elif test_id == 3:  # 调试随机抽选样本
        filename = 'loan.txt'
        data, featName = loadDataSet1(filename) # 测试最优特征选择
        choose_samples(data, k)  # 
        
 
    else:
        print('Wrong test_id!')