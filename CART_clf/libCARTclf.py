#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:24:51 2018

@author: suliang
"""

# CART_clf = CART classify

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadDataSets():
    data = [[1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    featName = ['no surfacing','flippers']
    return data, featName

class Node:  # 定义一个类，作为树的数据结构
    def __init__(self, feat = -1, value = None, results = None, right = None, left = None):
        self.feat = feat  # 列索引
        self.value = value  # 划分值
        self.results = results  # 所存储叶子结点所属类别
        self.right = right   # 右子树
        self.left = left    # 左子树


def label_uniq_cnt(data):    # 采用更简洁的counter写法进行技术
    from collections import Counter
    label = data[:,-1]  # 假定最后一列是label
    label_uniq_cnt = Counter(label)   # 使用Counter()函数，可以统计每个元素出现次数，返回list或dict
    return label_uniq_cnt
    

def cal_gini_index(data):
    m = data.shape[0]  # 样本数
    if data.shape[0] ==0:
        return 0
    label_counts = label_uniq_cnt(data)  # 取出不重复的数值
    
    gini = 0
    for label in label_counts:  # 取出每一个不重复的取值
        gini = gini + label_counts[label]**2      
    gini = 1 - float(gini) / pow(m, 2) # 计算每个取值的gini = 1-sum(p**2) 
    return gini


def split_tree(data, feat, value):
    set_1 = []
    set_2 = []
    for x in data:
        if x[feat] >=value:
            set_1.append(x)
        else:
            set_2.append(x)
    return (set_1, set_2)


def build_tree(data):  # 基于CART分类模型创建分类树
    data = np.array(data)
    if data.shape[0] == 0:   # 如果数据行数为0，则返回node
        return node
    
    currentGini = cal_gini_index(data) # 计算当前数据集的gini = 1-sum(p**2)
    bestGain = 0.0
    bestCriteria = None  # 元组存储(特征名称，最佳切分点)
    bestSets = None  # 存储切分后的数据子集(CART只会切分成左右2个子集)
    
    feature_num = data.shape[1] - 1  # 特征个数
    
    for feat in range(0, feature_num): # 外层循环，取出每一个特征
        feature_values = {}
        for sample in data:  # 内层循环，取出每一个样本
            feature_values[sample[feat]] = 1  # 取得该特征列所有可能的取值
            
        for value in feature_values.keys():
            (set_1, set_2) = split_tree(data, feat, value)
            
            nowGini = float(len(set_1)*cal_gini_index(set_1) + 
                            len(set_2)*cal_gini_index(set_2))/ len(data)
            gain = currentGini - nowGini
            
            if gain >bestGain and len(set_1)>0 and len(set_2) >0:
                bestGain = gain
                bestCriteria = (feat, value)
                bestSets = (set_1, set_2)
    
    if bestGain > 0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return node(feat=bestCriteria[0], value = bestCriteria[1], right=right, left = left)
    else:
        return node(results = label_uniq_cnt(data))
        
    
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


def choose_samples(data, k):
    m, n = np.shape(data)
    feature = []
    for j in range(k):
        feature.append(rd.randint(0, n-2))
    index = []
    for i in range(m):
        index.append(rd.randint(0, m-1))
    data_samples = []
    for i in range(m):
        data_temp = []
        for feat in feature:
            data_tmp.append(data[index[i]][feat])
        data_temp.append(data[index[i]][-1])
        data.samples.append(data_temp)
    return data_samples, feature


def test_simpleFishTree():
    node = Node()
    data, featName = loadDataSets()
    #labels = labels.reshape(-1,1)
    # color = labels*30+30
    #plt.scatter(data[:,0],data[:,1], c = color)
    myTree = build_tree(data)
    return data, featName
        
#------运行区-------------
if __name__ == '__main__':
    
    test_id = 0    # 程序运行前，需要指定test_id
    
    if test_id == 0:  # 调试生成一个简单的CART分类树
        data, featName = test_simpleFishTree()
    
    elif test_id == 1:
        test_buildTree()

    
    else:
        print('Wrong test_id!')        
    
    
    
    
    
    