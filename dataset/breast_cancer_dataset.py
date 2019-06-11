#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import load_breast_cancer

class BreastCancerDataset():
    """sklearn自带乳腺癌数据集(596,30)，共计596个样本，每个样本有30个特征，
    属于二分类数据集，分别为患乳腺癌(1)和正常(0)
    breask_cancer.keys() = ['data','target','target_names','DESCR','feature_names', 'filename']
    其中data(596,30), target(596,), 其中target中1代表患乳腺癌，0代表正常
    """
    def __init__(self):
        self.dataset = load_breast_cancer()
        self.datas = self.dataset.data      # (596, 30)
        self.labels = self.dataset.target   # (596,)
        self.feat_names = self.dataset.feature_names  # (30,)
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]
        
        return [data, label]

if __name__ == '__main__':
    bcset = BreastCancerDataset()
    data, label = bcset[10]                                      