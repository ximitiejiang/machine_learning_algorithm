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

# 从所有样本中有放回选出m x k的样本子集
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


#--------------------运行区-----------------------------------
if __name__ == '__main__':
    
    test_id = 1    # 程序运行前，需要指定test_id
    
    if test_id == 0:  # 调试choose_sample子程序
        k = 2
        x,labels = classifyData_1()
        data = np.hstack((x,labels.reshape(-1,1))) # 组合数据与标签
        plt.scatter(data[:,0], data[:,1], c=labels*30 + 30)        
        data_samples, feature = choose_samples(data, k)
        
    
    elif test_id == 1: 
        pass
    
    elif test_id == 2: # 完整调试
        x,labels = classifyData_1()
        data = np.hstack((x,labels.reshape(-1,1))) # 组合数据与标签
        randomForest_traing(data, 2)
    
    else:
        print('Wrong test_id!')