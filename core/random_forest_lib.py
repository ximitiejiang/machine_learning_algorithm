#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:59:21 2019

@author: suliang
"""

from decision_tree import CART

class RandomForest(CART):
    
    def __init__(self, feats, labels, n_trees):
        """随机森林分类算法特点：属于2中集成学习方法(bagging和boosting)中的bagging
        - bagging：代表是random forest，对数据集进行有放回抽样，形成多个训练数据的子集，并在每个子集训练一个分类器，
          然后多个分类器的结果投票决定预测结果。
        - boosting：代表是adboost和GBDT，对数据样本进行加权，并多轮训练一个基本分类器的分类误差率来更新训练数据集的权重，
          同时更新分类器的系数，最后对基本分类器进行线性组合得到最终的一个分类器。
        """
        super().__init__(feats, labels)
        self.n_trees = n_trees
        
    
    def train(self):
        
        self.tree_list = []
        for i in range(self.n_trees):
            feats = self.choose_samples()
            
            tree = self.create_tree()
            
            tree_list.append(tree)
        
        # 保存模型参数
        self.model_dict['model_name'] = 'RandomForest classifier'
        self.model_dict['tree_list'] = self.tree_list
    
    def choose_samples(self):
        pass
        
    def predict_single():
        pass
    
    