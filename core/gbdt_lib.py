#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:05:40 2019

@author: ubuntu
"""
from .decision_tree_lib import CARTReg
import numpy as np

class CrossEntropy():
    """交叉熵损失函数：
    loss = -(y*logy'+(1-y)*log(1-y'))
    grad = loss' = -(y/y' + (1-y)/(1-y'))
    """
    def loss(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

class MSE():
    """均方损失函数：
    loss = sum((y-y')^2) / n
    grad = loss' = 
    """
    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)
    
    def gradient(self, y, y_pred):
        return -(y - y_pred)
    
    

class GBDT(CARTReg):
    
    def __init__(self, n_clfs, feats, labels,    
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity_reduction = 1e-7):
        """GBDT梯度提升树算法特点：属于boost算法的一部分，所以总思路跟ada boost有类似地方，是串行训练多个模型，
        每个模型会在前一个模型基础上进一步优化，最后对串行的多个模型做累加集成。参考：https://www.cnblogs.com/zongfa/p/9314440.html
        - ada boost的串行优化体现在：前一个模型会根据误差率更新错分样本权重，从而影响串行下一个模型的选择;
        - 而GBDT的串行优化体现在：
        过程：
        1. 创建回归树
        """
        super().__init__(feats, labels,    
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity_reduction = 1e-7)
        # assert labels are one hot labels 
        assert self.labels.ndim >= 2, 'the labels should be one-hot labels.'

        self.n_clfs = n_clfs
        self.loss = CrossEntropy()
        
        # 创建n个分类器
        self.clf_list = []
        y_pred = np.ones(self.feats.shape) * self.feats.mean()  # 首轮用均值作为预测值初始化
        for clf_id in range(self.n_clfs):
            residual = -self.loss.gradient()   # 用损失函数的负梯度作为残差的近似
            tree = self.create_tree(self, feats, labels, current_depth=0)  #
        
        # 保存参数
        self.model_dict['model_name'] = 'GBDT' + '_'  \
                                        + str(self.n_clfs) \
                                        + 'clfs'
        self.model_dict['clf_list'] = self.clf_list
    
    def predict_single(self):
        pass
