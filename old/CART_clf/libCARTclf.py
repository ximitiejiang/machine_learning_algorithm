#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 10:42:28 2018

@author: suliang
"""

''' new function for CART_classify algorithm
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadDataSet1(filename):  # 用来加载<统计学习方法>中的贷款申请数据集
    fr = open(filename)
    data = [inst.strip().split('\t') for inst in fr.readlines()]
    featName = ['age', 'job', 'house','credit']
    return data, featName


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


# -------------------运行区-------------------------------------------
if __name__ == '__main__':
    
    test_id = 1    # 程序运行前，需要指定test_id
    
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
 
    else:
        print('Wrong test_id!')
        
        