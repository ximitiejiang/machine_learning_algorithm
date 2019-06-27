#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:37:06 2019

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt
from dataset.loan_dataset import LoanDataset

class Node:
    def __init__(self, fea=-1, value=None, results=None, right=None, left=None):
        self.fea = fea  # 用于切分数据集的属性的列索引值
        self.value = value  # 设置划分的值
        self.results = results  # 存储叶节点的值
        self.right = right  # 右子树
        self.left = left  # 左子树
    

class CART():
    
    def __init__(self, data):
        
        
        self.tree = self.create_tree(data)
    
    def create_tree(self, data):
        '''构建树
        input:  data(list):训练样本
        output: node:树的根结点
        '''
        # 构建决策树，函数返回该决策树的根节点
        if len(data) == 0:
            return Node()
        
        # 1、计算当前的Gini指数
        currentGini = self.cal_gini_index(data)
        
        bestGain = 0.0
        bestCriteria = None  # 存储最佳切分属性以及最佳切分点
        bestSets = None  # 存储切分后的两个数据集
        
        feature_num = len(data[0]) - 1  # 样本中特征的个数
        # 2、找到最好的划分
        for fea in range(0, feature_num):
            # 2.1、取得fea特征处所有可能的取值
            feature_values = {}  # 在fea位置处可能的取值
            for sample in data:  # 对每一个样本
                feature_values[sample[fea]] = 1  # 存储特征fea处所有可能的取值
            
            # 2.2、针对每一个可能的取值，尝试将数据集划分，并计算Gini指数
            for value in feature_values.keys():  # 遍历该属性的所有切分点
                # 2.2.1、 根据fea特征中的值value将数据集划分成左右子树
                (set_1, set_2) = self.split_tree(data, fea, value)
                # 2.2.2、计算当前的Gini指数
                nowGini = float(len(set_1) * self.cal_gini_index(set_1) + \
                                 len(set_2) * self.cal_gini_index(set_2)) / len(data)
                # 2.2.3、计算Gini指数的增加量
                gain = currentGini - nowGini
                # 2.2.4、判断此划分是否比当前的划分更好
                if gain > bestGain and len(set_1) > 0 and len(set_2) > 0:
                    bestGain = gain
                    bestCriteria = (fea, value)
                    bestSets = (set_1, set_2)
        
        # 3、判断划分是否结束
        if bestGain > 0:
            right = self.create_tree(bestSets[0])
            left = self.create_tree(bestSets[1])
            return Node(fea=bestCriteria[0], value=bestCriteria[1], \
                        right=right, left=left)
        else:
            return Node(results=self.label_uniq_cnt(data))  # 返回当前的类别标签作为最终的类别标签
    
    def label_uniq_cnt(self, data):
        '''统计数据集中不同的类标签label的个数
        input:  data(list):原始数据集
        output: label_uniq_cnt(int):样本中的标签的个数
        '''
        label_uniq_cnt = {}
        
        for x in data:
            label = x[len(x) - 1]  # 取得每一个样本的类标签label
            if label not in label_uniq_cnt:
                label_uniq_cnt[label] = 0
            label_uniq_cnt[label] = label_uniq_cnt[label] + 1
        return label_uniq_cnt
    
    def cal_gini_index(self, data):
        '''计算给定数据集的Gini指数
        input:  data(list):树中
        output: gini(float):Gini指数
        '''
        total_sample = len(data)  # 样本的总个数 
        if len(data) == 0:
            return 0   
        label_counts = self.label_uniq_cnt(data)  # 统计数据集中不同标签的个数
        
        # 计算数据集的Gini指数
        gini = 0
        for label in label_counts:
            gini = gini + pow(label_counts[label], 2)
            
        gini = 1 - float(gini) / pow(total_sample, 2)
        return gini
    
    def split_tree(self, data, fea, value):
        '''根据特征fea中的值value将数据集data划分成左右子树
        input:  data(list):数据集
                fea(int):待分割特征的索引
                value(float):待分割的特征的具体值
        output: (set1,set2)(tuple):分割后的左右子树
        '''
        set_1 = []
        set_2 = []
        for x in data:
            if x[fea] >= value:
                set_1.append(x)
            else:
                set_2.append(x)
        return (set_1, set_2)

    
    def predict(self, sample, tree):
        '''对每一个样本sample进行预测
        input:  sample(list):需要预测的样本
                tree(类):构建好的分类树
        output: tree.results:所属的类别
        '''
        # 1、只是树根
        if self.tree.results != None:
            return self.tree.results
        else:
        # 2、有左右子树
            val_sample = sample[self.tree.fea]
            branch = None
            if val_sample >= self.tree.value:
                branch = self.tree.right
            else:
                branch = self.tree.left
            return self.predict(sample, branch)
    
    
class ID3():
    def __init__(self):
        pass
    
class C45():
    def __init__(self):
        pass

if __name__ == '__main__':
    
    dataset = 'loan'
    
    if dataset == 'treedata':
        data = []
        with open('./dataset/simple/treedata.txt') as f:
            for line in f.readlines():
                sample = []
                lines = line.strip().split("\t")
                for x in lines:
                    sample.append(float(x))  # 转换成float格式
                data.append(sample)
        data = np.array(data)        # (200, 2)
    
        plt.scatter(data[:,0],data[:,1])
        cart = CART(data)
        cart.predict([1,1], cart.tree)
    
    if dataset == 'loan':
        dataset = LoanDataset()
        x = dataset.datas
        y = dataset.labels
        data = np.concatenate([x,y.reshape(-1,1)], axis=1)
        
        
        cart = CART(data)
        cart.predict([1,1], cart.tree)
        