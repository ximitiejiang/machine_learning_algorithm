#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""


import matplotlib.pyplot as plt
from dataset.multi_class_dataset import MultiClassDataset
from core.kd_tree_lib import KdTree

if __name__ == "__main__":
    
    source = 'multi'
    
    if source == 'multi':
        dataset = MultiClassDataset(n_samples=100, centers=2, n_features=2)
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        knn = KdTree(x, y)

