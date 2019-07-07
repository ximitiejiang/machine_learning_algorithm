#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets.samples_generator import make_blobs
from .base_dataset import BaseDataset

class MultiClassDataset(BaseDataset):
    """sklearn自带多分类数据集，可生成2-n任意类别数据集，基本是线性可分，但有可能部分点线性不可分
    
    """
    def __init__(self, n_samples=100, centers=3, n_features=4, 
                 center_box=(-10.0,10.0),
                 cluster_std=1.0,
                 norm=None, 
                 label_transform_dict=None, 
                 one_hot=None,
                 binary=None,
                 shuffle=None):
        
        self.n_samples = n_samples
        self.centers = centers
        self.n_features = n_features
        self.center_box = center_box
        self.cluster_std = cluster_std
        super().__init__(norm=norm, 
                         label_transform_dict=label_transform_dict, 
                         one_hot=one_hot,
                         binary=binary,
                         shuffle=shuffle)
        
    def get_dataset(self):
        dataset = {}
        datas, labels = make_blobs(n_samples=self.n_samples, 
                                     centers=self.centers, 
                                     n_features=self.n_features,
                                     center_box = self.center_box,
                                     cluster_std = self.cluster_std)         # 可设置中心点范围和方差，这里用默认值
        dataset['data'] = datas
        dataset['target'] = labels
        return dataset

               