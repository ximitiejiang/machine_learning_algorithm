#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:31:12 2019

@author: ubuntu
"""
from .decision_tree_lib import BaseTree
import numpy as np

class LogisticLoss():
    """逻辑损失函数：
    loss = y*logy'+(1-y)*log(1-y')
    grad = loss' = -(y/y' - (1-y)/(1-y'))
    """
    def loss(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        y_pred = self.sigmoid(y_pred)
        return y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)  # 逻辑损失跟交叉熵相比缺了个负号

    def gradient(self, y, y_pred): # 注意这里的梯度根据xgboost的公式是对y_pred求导，梯度的求解公式似乎把负号又纠正过来了
        y_pred = self.sigmoid(y_pred)
        return -(y - y_pred)

    def sigmoid(self, x): # 概率化到[0,1]之间
        return 1 / (1 + np.exp(-x))
    
    def hess(self, y, y_pred):   # 
        y_pred = self.sigmoid(y_pred)
        return y_pred * (1 - y_pred)

class XGBoost(BaseTree):
    
    def __init__(self, feats, labels,
                 n_clfs, learning_rate=0.5, 
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity_reduction = 1e-7):
        assert self.labels.ndim >= 2, 'the labels should be one-hot labels.' # 由于采用多分类交叉熵做损失函数，需要采用标签独热编码(即概率化标签)
        """XGBoost极端梯度增强算法：
        """
        super().__init__(feats=feats, 
                         labels=labels,    
                         min_samples_split=min_samples_split, 
                         max_depth=max_depth,
                         min_impurity_reduction = min_impurity_reduction)
        self.n_clfs = n_clfs
        self.learning_rate = learning_rate
        self.loss = LogisticLoss()
        self.clf_list = []
    
    def train():
        pass
    
    def gain_by_taylor(self, labels):
        """决定数据分割节点：定义一个gain，类似基尼缩减，熵缩减(信息增益)，都是越大越好，这里gain也是越大越好"""
        pass
    
    def calc_gain(self):
        pass
    
    def appromimate_update(self, labels):
        """决定叶子节点的取值为残差？"""
        pass