#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import load_breast_cancer
from .base_dataset import BaseDataset

class BreastCancerDataset(BaseDataset):
    """sklearn自带乳腺癌数据集(596,30)，共计596个样本，每个样本有30个特征，
    属于二分类数据集，分别为患乳腺癌(1)和正常(0)
    radius: 半径
    texture: 纹理，灰度值的标准偏差
    perimeter: 周长
    area: 面积
    smoothness: 平滑度，半径的变化幅度
    compactness: 密实度，周长的平方除以面积 - 1 
    concavity: 凹度，凹陷的部分轮廓严重程度
    concave points: 凹点，凹陷的轮廓数量
    symmetry: 对称性
    fractal dimension: 分形维度
        
    breask_cancer.keys() = ['data','target','target_names','DESCR','feature_names', 'filename']
    其中data(596,30), target(596,), 其中target中1代表患乳腺癌，0代表正常
    """
    def __init__(self, 
                 norm=None, label_transform_dict=None, one_hot=None, binary=None, shuffle=None):
        super().__init__(norm=norm, 
                         label_transform_dict=label_transform_dict, 
                         one_hot=one_hot,
                         binary=binary,
                         shuffle=shuffle)
    
    def get_dataset(self):
        return load_breast_cancer()
              