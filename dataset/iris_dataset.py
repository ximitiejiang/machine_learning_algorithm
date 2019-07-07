#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import load_iris
from .base_dataset import BaseDataset

class IrisDataset(BaseDataset):
    """sklearn自带鸢尾花数据集，共计150个样本，每个样本有4个特征，
    属于多分类数据集，共有3个类别
    setosa
    versicolor
    virginica
        
    iris.keys() = ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
    其中data(150,4), target(150,)
    """
    def __init__(self, norm=None, label_transform_dict=None, one_hot=None, 
                 binary=None, shuffle=None):
        super().__init__(norm=norm, 
                         label_transform_dict=label_transform_dict, 
                         one_hot=one_hot,
                         binary=binary,
                         shuffle=shuffle)
    
    def get_dataset(self):
        return load_iris()
    
            