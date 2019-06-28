#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_circles
from .base_dataset import BaseDataset

class NonlinearDataset(BaseDataset):
    """sklearn自带非线性数据集，可生成二分类的非线性可分数据集
    有两种数据集可选，一种moon月亮形，一种circle圆圈形
    """
    def __init__(self, type= 'moon', n_samples=100, noise=0.1):
        self.type = type
        self.n_samples = n_samples
        self.noise = noise
        super().__init__()
        
            
    def get_dataset(self):
        if self.type == 'moon':
            datas, labels = make_moons(n_samples=self.n_samples, 
                                                 noise=self.noise)
        elif self.type == 'circle':
            datas, labels = make_circles(n_samples=self.n_samples, 
                                                   noise=self.noise)
        else:
            print('wrong type input.')
        
        dataset = {}
        dataset['data'] = datas
        dataset['target'] = labels
        return dataset
    
