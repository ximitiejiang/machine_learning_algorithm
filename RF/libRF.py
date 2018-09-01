#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:24:51 2018

@author: suliang
"""

# RF = Random Forest

def loadDataSets():
    pass


class node:  # 定义一个类，作为树的数据结构
    def __init__(self, fea = -1, value = None, results = None, right = None, left = None):
        self.fea = fea  # 列索引
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


def split_tree():
    pass


def build_tree(data):  # 基于CART分类模型创建分类树
    if len(data) == 0:   # 如果
        return node
    currentGini = cal_gini_index(data)
    
    bestGain = 0.0
    bestCriteria = None
    bestSets = NOne
    
    feature_num = len(data[0]) - 1
    
    for fea in range(0, feature_num):
        feature_values = {}
        for sample in data:
            feature_values[sample[fea]] = 1
            
        for value in feature_values.keys():
            (set_1, set_2) = split_tree(data, fea, value)
            
            nowGini = float(len(set_1)*cal_gini_index(set_1) + 
                            len(set_2)*cal_gini_index(set_2))/ len(data)
            gain = currentGini - nowGini
            
            if gain >bestGain and len(set_1)>0 and len(set_2) >0:
                bestGain = gain
                bestCriteria = (fea, value)
                bestSets = (set_1, set_2)
    
    if bestGain > 0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return node(fea=bestCriteria[0], value = bestCriteria[1], right=right, left = left)
    else:
        return node(results = label_uniq_cnt(data))
        
    
def predict():
    pass

#------test-------------
def test_buildTree():
    loadDataSets()
    build_tree()
        
#------运行区-------------
test_buildTree()
    
    
    
    