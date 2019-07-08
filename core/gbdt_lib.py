#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:05:40 2019

@author: ubuntu
"""
from .decision_tree_lib import BaseTree, CARTReg
import numpy as np
import time

class CrossEntropy():
    """交叉熵损失函数：
    loss = -(y*logy'+(1-y)*log(1-y'))
    grad = loss' = -(y/y' - (1-y)/(1-y'))
    """
    def loss(self, y, y_pred):  # loss一般为正值，代表数据集的熵
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

    def gradient(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y / y_pred) + (1 - y) / (1 - y_pred)
    
    def acc(self, y, y_pred):
        acc = np.sum(y == y_pred, axis=0) / len(y)
        return acc


class MSE():
    """均方损失函数：
    loss = sum((y-y')^2) / n
    grad = loss' = 
    """
    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)
    
    def gradient(self, y, y_pred):
        return -(y - y_pred)
    
    

class GBDT(BaseTree):
    
    def __init__(self, feats, labels,
                 n_clfs, learning_rate=0.5, 
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
        super().__init__(feats=feats, 
                         labels=labels,    
                         min_samples_split=min_samples_split, 
                         max_depth=max_depth,
                         min_impurity_reduction = min_impurity_reduction)
        assert self.labels.ndim >= 2, 'the labels should be one-hot labels.' # 由于采用多分类交叉熵做损失函数，需要采用标签独热编码(即概率化标签)

        self.n_clfs = n_clfs
        self.learning_rate = learning_rate
        self.loss = CrossEntropy()
        self.clf_list = []
     
    def train(self):    
        # 创建n个分类器
        y_pred = np.ones(self.labels.shape) * self.labels.mean()     # 首轮用均值作为预测值初始化
        for clf_id in range(self.n_clfs):
            residual = -self.loss.gradient(self.labels, y_pred)      # 用损失函数的负梯度作为残差的近似: 对回归树采用分类的交叉熵损失函数，是因为本质上是评估分类结果而不是回归结果。
            clf = CARTReg(self.feats, residual,    
                          min_samples_split=self.min_samples_split, 
                          max_depth=self.max_depth,
                          min_impurity_reduction = 1e-7).train()     # 基于残差创建回归树模型
            residual_pred = clf.evaluation(self.feats, residual)     # 
#            self.tree = self.create_tree(self.feats, residual)       # 创建基于残差的一棵回归树：为了让evaluation工作，需要对象有self.tree属性
#            residual_pred = super().evaluation(self.feats, residual) # 输入的是残差构建一棵树，所以预测的也是残差
            y_pred += np.multiply(self.learning_rate, residual_pred)  # 残差乘以学习率后累加 (注意如果前面残差不是用负梯度而是用梯度做近似，则这里就是累减而不是累加)
            self.clf_list.append(clf)
        
        # 保存参数
        self.trained = True
        self.model_dict['model_name'] = 'GBDT' + '_'  \
                                        + str(self.n_clfs) \
                                        + 'clfs'
        self.model_dict['clf_list'] = self.clf_list
        return self
    

    
    def predict_single(self, sample):
        """仅实现分类模型的单样本预测，沿用父类的分类evaluation"""
        y_pred = np.array([])
        for clf in self.clf_list:
            residual_pred = clf.predict_single(sample) * self.learning_rate  
            y_pred = residual_pred if not y_pred.any() else y_pred + residual_pred
        # softmax概率化
        y_pred = self.softmax(y_pred)
        y_pred = np.argmax(y_pred)
        return y_pred
        
    def softmax(self, y):
        y_exp = np.exp(y)
        prob = y_exp / sum(y_exp)
        return prob
    
