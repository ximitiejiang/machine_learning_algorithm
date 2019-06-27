#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import load_boston
from .base_dataset import BaseDataset
import numpy as np

data = {'age':    [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
        'job':    [2,2,1,1,2,2,2,1,2,2,2,2,1,1,2],
        'house':  [2,2,2,1,2,2,2,1,1,1,1,1,2,2,2],
        'credit': [3,2,2,3,3,3,2,2,1,1,1,2,2,1,3],
        'approve':[0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]}


class LoanDataset(BaseDataset):
    """loan数据集(506,13)，来自lihang的统计学习方法P59，属于离散数据集
    下面特征值取值一般是越小越好（小的1为青年，有工作，有房子，信贷好）
    age: 1为青年，2为中年，3为老年
    job: 1为有工作，2为没工作
    house: 1为有房子，2为没房子
    credit：1为信贷非常好，2为信贷好，3为信贷一般
    
    approve：该列为label，0为不批准贷款，1为批准贷款
    """
    def __init__(self):
        super().__init__()
    
    def get_dataset(self):
        feat_names = []
        feats = []
        for key, value in data.items():
            feat_names.append(key)
            feats.append(np.array(value))
        feats = np.stack(feats, axis=0).T
        
        dataset = {}
        dataset['data'] = feats[:, :-1]
        dataset['target'] = feats[:, -1]
        dataset['target_names'] = feat_names
            
        return dataset

    
               