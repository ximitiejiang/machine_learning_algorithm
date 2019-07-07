#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:05:40 2019

@author: ubuntu
"""
from .decision_tree_lib import CARTReg
import numpy as np
import time

class CrossEntropy():
    """交叉熵损失函数：
    loss = -(y*logy'+(1-y)*log(1-y'))
    grad = loss' = -(y/y' - (1-y)/(1-y'))
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
        self.calc_leaf_value = self.mean_of_y
        self.calc_impurity_reduction = self.calculate_variance_reduction
        
        # 创建n个分类器
        y_pred = np.ones(self.labels.shape) * self.labels.mean()       # 首轮用均值作为预测值初始化
        for clf_id in range(self.n_clfs):
            residual = -self.loss.gradient(self.labels, y_pred)      # 用损失函数的负梯度作为残差的近似: 对回归树采用分类的交叉熵损失函数，是因为本质上是评估分类结果而不是回归结果。
            self.tree = self.create_tree(self.feats, residual)       # 创建基于残差的一棵回归树：为了让evaluation工作，需要对象有self.tree属性
            residual_pred = super().evaluation(self.feats, residual) # 输入的是残差构建一棵树，所以预测的也是残差
            y_pred += np.multiply(self.learning_rate, residual_pred) # 残差乘以学习率后累加 (注意如果前面残差不是用负梯度而是用梯度做近似，则这里就是累减而不是累加)
            self.clf_list.append(self.tree)
        
        # 保存参数
        self.trained = True
        self.model_dict['model_name'] = 'GBDT' + '_'  \
                                        + str(self.n_clfs) \
                                        + 'clfs'
        self.model_dict['clf_list'] = self.clf_list
        return self
    
    def predict_single(self, sample):
        y_pred = np.array([])
        for clf in self.clf_list:
            self.tree = clf   # 基于回归树的模型填入self.tree，从而可以使用回归树的evalluation方法
            residual_pred = super().predict_single(sample)
            y_pred += residual_pred
    
    def evaluation(self, test_feats, test_labels, show=False):
        """单样本预测的predict_single对分类树和回归树都一样，而全样本评估的evaluation需要改，
        是因为gbdt继承的是回归树，但需要用分类树的evaluation(未作修改)
        Args:
            test_feats: (n_sample, n_feat)
            test_labels: (n_sample,)
        """       
        correct = 0
        total_sample = len(test_feats)
        start = time.time()
        for feat, label in zip(test_feats, test_labels):
            pred_label = self.predict_single(feat)
            if int(pred_label) == int(label):
                correct += 1
        acc = correct / total_sample
        print('======%s======'%self.model_dict['model_name'])
        print('Finished evaluation in %f seconds with accuracy = %f.'%((time.time() - start), acc))
        
        return acc
