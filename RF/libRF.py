#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:24:51 2018

@author: suliang
"""

# RF = Random Forest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadDataSets():
    from sklearn.datasets.samples_generator import make_classification
    X,labels=make_classification(n_samples=200,n_features=2,n_redundant=0,
                                 n_informative=2, random_state=1,
                                 n_clusters_per_class=2)
    rng=np.random.RandomState(2)
    X+=2*rng.uniform(size=X.shape)
    return X, labels

class Node:  # 定义一个类，作为树的数据结构
    def __init__(self, feat = -1, value = None, results = None, right = None, left = None):
        self.feat = feat  # 列索引
        self.value = value  # 划分值
        self.results = results  # 所存储叶子结点所属类别
        self.right = right   # 右子树
        self.left = left    # 左子树


def label_uniq_cnt(data):
    label_uniq_cnt = {}
    for x in data:
        label = x[len(x) - 1]
        if label not in label_uniq_cnt:
            label_uniq_cnt[label] = 0
        label_uniq_cnt[label] = label_uniq_cnt[label] + 1
    return label_uniq_cnt
    

def cal_gini_index(data):
    total_sample = len(data)
    if len(data) ==0:
        return 0
    label_counts = label_uniq_cnt(data)
    
    gini = 0
    for label in label_counts:
        gini = gini + pow(label_counts[label], 2)
    
    gini = 1 - float(gini) / pow(total_sample, 2)
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
    if len(data) == 0:   # 如果数据行数为0，则返回node
        return node
    
    currentGini = cal_gini_index(data) # 计算当前数据集的gini
    bestGain = 0.0
    bestCriteria = None
    bestSets = None
    
    feature_num = len(data[0]) - 1
    
    for feat in range(0, feature_num): # 外层循环，取出每一个特征
        feature_values = {}
        for sample in data:  # 内层循环，取出每一个样本
            feature_values[sample[feat]] = 1
            
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




#------test-------------
def test_buildTree():
    node = Node()
    X, labels = loadDataSets()
    #labels = labels.reshape(-1,1)
    color = labels*30+30
    plt.scatter(X[:,0],X[:,1], c = color)
    
    data = np.hstack((X,labels.reshape(-1,1)))
    myTree = build_tree(data)
    
    sample = [1,-1]
    result = predict(sample ,myTree)
        
#------运行区-------------
myTree = test_buildTree()
    
    
    
    