#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:05:40 2019

@author: ubuntu
"""
from .decision_tree_lib import CARTReg

class CrossEntropy():
    
    def loss(self):
        pass
    
    def gradient(self):
        pass


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
        self.n_clfs = n_clfs
        self.loss = CrossEntropy()
        
        # 创建n个分类器
        self.clf_list = []
        for clf_id in range(self.n_clfs):
            residual = self.loss.gradient()   # 用损失函数的梯度作为残差的近似
            tree = self.create_tree(self, feats, labels, current_depth=0)
        
        # 保存参数
        self.model_dict['model_name'] = 'GBDT' + '_'  \
                                        + str(self.n_clfs) \
                                        + 'clfs'
        self.model_dict['clf_list'] = self.clf_list
    
    def predict_single(self):
        pass
