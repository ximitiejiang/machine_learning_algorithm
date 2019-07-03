#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:08:27 2019

@author: ubuntu
"""
from .base_model import BaseModel

class DTStump:
    """DTStump代表decision tree stump也就是决策树树桩(树桩代表只有一个root和2个leaf)"""
    def __init__(self, feat_id=None, feat_value=None, polarity=1, alpha=None):
        self.feat_id = feat_id
        self.feat_value = feat_value  
        self.polarity = polarity   # 存放预测样本正负的方式(polarity=1则小于value为负样本大于value为正样本)
        self.alpha = alpha        # 该分类器的权重：误差越小，alpha越大，对应该分类器在预测中越重要

class LogisticStump:
    """尝试用logistic reg分类器来做基础弱分类器(需要确保弱分类器的分类误差率大于0.5)"""
    def __init__():
        pass

CLF_dict = {'DTStump': DTStump,
            'LogisticStump': LogisticStump}

class AdaBoost(BaseModel):
    
    def __init__(self, feats, labels, n_clfs, clf_type='DTStump'):
        """Ada boost分类器算法特点：采用最简单的只有一层的树作为基础分类器
        (也就是depth=1, root只进行了一次分解得到2个叶子节点)
        """
        super().__init__(feats, labels)
        
        self.n_clfs = n_clfs
        self.clf_list = []
        CLF = CLF_dict[clf_type]
        # 创建n个分类器
        for _ in range(self.n_clfs):
            
            for feat_id in range(self.n_feats):  # 遍历每个特征的每个特征值
                feat = feats[:, feat_id]
                feat_unique = np.unique(feat)
                for value in feat_unique:
                    
                    # TODO: 查看
                    
    def predict_single(self):
        
        