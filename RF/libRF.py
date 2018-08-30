#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:24:51 2018

@author: suliang
"""

# RF = Random Forest

class node:
    def __init__(self, fea = -1, value = None, results = None, right = None, left = None):
        self.fea = fea  # 列索引
        self.value = value  # 划分值
        self.results = results  # 所存储叶子结点所属类别
        self.right = right   # 右子树
        self.left = left    # 左子树

def build_tree():
    if len(data) == 0:
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
        


        
        