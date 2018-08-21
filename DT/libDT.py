#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 11:57:50 2018

@author: suliang
"""
import numpy as np

def creatDataSets():
    # 这个数据集是一个鱼特征数据集，第一列代表no surfacing特征, 第二列代表flippers特征
    # 第三列为labels列，yes代表是鱼，no代表不是鱼
    data = [[1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    featName = ['no surfacing',
              'flippers']
    return data, featName


def calcShannonEnt(data):  # 计算一个数据集的信息熵
    from math import log
    # 计数
    numEntries = len(data)
    labelcounts = {}  # 字典用于放置{总计多少类：每个类多少样本个数}
    for line in data: # 循环取每一行样本
        currentlabel = line[-1]
        if currentlabel not in labelcounts.keys(): # 如果是一个新分类
            labelcounts[currentlabel] = 0       # 在字典新增这个新分类，对应样本个数0
        labelcounts[currentlabel] +=1    # 如果不是新分类： 在旧类的样本个数+1
    
    # 计算一个数据集的信息熵
    shannonEnt = 0.0
    for key in labelcounts:
        pi = float(labelcounts[key])/numEntries # 计算每一个分类的概率p=分类数/总数
        shannonEnt -= pi * log(pi,2)    # 计算entropy = -sum(p*logp)
    return shannonEnt


def splitDataSet(data, axis, value):  
    # 划分数据集，输入数据集data, 特征列号axis，对应特征列的特征值
    # 比如splitDataSet(data,0,1)代表第0列特征值为1的数据子集
    subDataSet = []
    for line in data:
        if line[axis] == value:
            reducedFeatVec = line[:axis]    # 取出axis列左的数据(不包括axis列)
            reducedFeatVec.extend(line[axis+1:]) # 取出axis列右的数据(不包括axis列)
            
            subDataSet.append(reducedFeatVec) 
            # extend是在一个元素内操作，append是在大的list操作不同元素
    return subDataSet   # 返回指定划分的子数据集


# 几个重要概念需要澄清：
# 1. 信息熵： 是针对一个数据集来说的，跟每类(标签概率p)相关
# 2. 子集权重熵之和：就是特征划分后各子集权重熵之和，跟子集熵和子集的(取值概率p)相关
# 3. 信息增益：是针对某一个特征来说的，就是(母数据集熵)与(某子集的权重熵)的差值
#    信息增益也就是某一特征划分的降熵能力，所以是(母数据集信息熵)减去(各子集权重熵之和)

def chooseBestFeatureToSplit(data):  
    numFeatures = len(data[0]) - 1   # 获得特征个数，label不算所以减一
    baseEntropy = calcShannonEnt(data)
    bestInfoGain = 0.0
    bestFeature = -1
    
    for i in range(numFeatures):  # 外循环定义特征i
        featList = [sample[i] for sample in data] #该句是获得每个特征的所有值，即i列
        uniqueVals = set(featList)  # set函数是获得不重复的值（即去除重复）
        
        newEntropy = 0.0
        for value in uniqueVals:  # 内循环定义该特征i的取值种类
            subDataSet = splitDataSet(data, i, value)           
            p = len(subDataSet)/float(len(data))  
            newEntropy += p * calcShannonEnt(subDataSet) # 计算划分子数据集的熵
        
        infoGain = baseEntropy - newEntropy # 计算特征i划分的信息增益
        if (infoGain > bestInfoGain): # 如果特征i的增益最大，则以该特征i为最优特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCount(classList):  # 该子程序只用来统计当剩下最后一列时，最多label出现次数
    import operator
    
    classCount = {}
    for vote in classList:    # 经典方法：用字典记录统计结果 {类型：出现次数}
        if vote not in classCount.keys():  # 如果是新类型，新增key，次数为0
            classCount[vote] = 0        
        classCount[vote] += 1              # 如果不是新类型，次数+1
    
    # 对统计结果的字典数据进行排序: 从大到小
    # 排序函数sorted(dict.iteritems(), )
    sortedClassCount = sorted(classCount.iteritems(), \
                              key = operator.itemgetter(1), reverse =True)
    return sortedClassCount[0][0]
        

def createTree(data, featName):
    featColumnName = featName[:]  # 在函数体内修改了featName,为了防止对体外同名变量的影响，
    classList = [example[-1] for example in data] # classlist为所有的标签值
    # 这种取list data标签的方式还真是累，之前就把数据处理成array，后边处理不是更方便？
    
    if classList.count(classList[0])==len(classList):  # 如果所有标签值相同，就返回
        return classList[0]
    if len(data[0])==1:   # 如果是遍历到了最后，data[0]就是
        return majorityCount(classList)  # 就返回次数最多的分类值
    
    bestFeat = chooseBestFeatureToSplit(data)
    bestfeatName = featColumnName[bestFeat]
    
    myTree = {bestfeatName:{}}
    del(featColumnName[bestFeat])  # 已经split过的特征名称就去掉,就是此处修改了形参
    
    featValues = [example[bestFeat] for example in data] # 获得最佳特征的列
    uniqueValues = set(featValues)  # 去除重复值
    for value in uniqueValues:  #
        subLabels = featColumnName[:]
        
        # 这句是整个createTree的核心：循环创建 特征树(即以特征为key，特征数值为value的字典)
        # 该特征树用字典表示如下：
        # {特征名称：{特征数值1:{下一棵子树},特征数值2:{下一棵子树}}}
        myTree[bestfeatName][value] = \
        createTree(splitDataSet(data,bestFeat,value), subLabels)
    
    return myTree


def classify(myTree, featName, testVec):  # 决策树用于分类
    firstName = list(myTree.keys())[0]  # myTree是一个字典，她的keys属性不能直接切片，需要转成list
    
    secondDict = myTree[firstName]
    featIndex = featName.index(firstName)
    
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featName, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel    


# ------main-----------
data, featName = creatDataSets()  # 创建数据

myTree = createTree(data, featName)  # 创建一棵树

result = classify(myTree, featName, [1,1])  # 基于已有一棵树进行新数据的分类



    
