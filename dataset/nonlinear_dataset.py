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
    def __init__(self, type= 'moon', n_samples=100, noise=0.1, label_transform_dict=None):
        self.type = type
        self.n_samples = n_samples
        self.noise = noise
        super().__init__()
        
        # 增加标签处理代码
        self.label_transform(label_transform_dict)
        
    def label_transform(self, label_transform_dict):
        """默认不改变label的取值范围，但可以通过该函数修改labels的对应范围
        例如svm需要label为[-1,1]，则可修改该函数。
        """
        if label_transform_dict is None:
            pass
        else:  # 如果指定了标签变换dict
            assert isinstance(label_transform_dict, dict), 'the label_transform_dict should be a dict.' 
            for i, label in enumerate(self.labels):
                new_label = label_transform_dict[label]
                self.labels[i] = int(new_label)   # 比如{1:1, 0:-1}就是要把1变为1, 0变为-1
            
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
    
