#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 07:35:43 2019

@author: ubuntu
"""

import numpy as np
from sklearn.preprocessing import scale
from .base_model import BaseModel
import time

class NaiveBayes(BaseModel):
    def __init__(self, feats, labels):
        """ naive bayes algorithm lib, 朴素贝叶斯模型
        参考：https://blog.csdn.net/u013597931/article/details/81705718 
        特点：没有参数，不需要训练，支持多分类，不仅支持连续性特征，而且支持离散型特征(离散型特征最适合朴素贝叶斯)
        
        Args:
            feats(numpy): (n_samples, f_feats)
            labels(numpy): (n_samples,)
        """
        super().__init__(feats, labels)
        
        class_mean_list, class_std_list = self.get_mean_std()  # (n_class, )()

        
    def predict_single(self):
        pass
    
    def evaluation(self, test_feats, test_labels):
        pass
    
    def get_mean_std(self):
        """获得训练集的均值和方差，
        已知self.n_classes, self.classes_list, self.n_samples, self.n_feats
        """
        # 按类别分解feats
        c_feats = [] 
        for i in range(self.n_classes):
            temp = self.feats[self.labels==self.classes_list[i]]
            c_feats.append(temp)
        #对每类特征的每种特征计算mean, std
        for i in range(self.n_classes):
            for j in range(self.n_feats):
                pass
    
    def d 
        