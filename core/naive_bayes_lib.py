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
        特点：支持多分类
        
        Args:
            feats(numpy): (n_samples, f_feats)
            labels(numpy): (n_samples,)
        """
        super().__init__(feats, labels)
        
        # assert labels are [-1, 1]
        label_set = set(self.labels)
        for label in label_set:
            if label != 1 and label != -1:
                raise ValueError('labels should be 1 or -1.')
        
    def predict_single(self):
        pass
    
    def get_mean_std(self):
        