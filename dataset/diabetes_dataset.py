#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import load_diabetes
from .base_dataset import BaseDataset

class DiabetesDataset(BaseDataset):
    """sklearn自带糖尿病数据集，共计442个样本，每个样本有10个特征，标签labels为一组患病指标数据，为连续性数据。
    属于回归数据集。
    特征名称
        age
        sex
        bmi bmi体质指数
        bp 平均血压
        s1,s2,s3,s4,s5,s6 一年后疾病指数
    
        
    diabetes.keys() = ['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename']
    其中data(442, 10), target(442,)
    """
    def __init__(self):
        super().__init__()
    
    def get_dataset(self):
        return load_diabetes()
    
            