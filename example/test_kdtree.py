#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""

import numpy as np
from dataset.multi_class_dataset import MultiClassDataset
from core.kd_tree_lib import KdTree

if __name__ == "__main__":
    
    source = 'multi'
    
    if source == 'test':  # points from Lihang's book(P42)
        x = np.array([[2., 3.],
                      [5., 4.],
                      [9., 6.],
                      [4., 7.],
                      [8., 1.],
                      [7., 2.]])
        y = np.array([1,1,0,1,0,0])    
        kd = KdTree(x,y)
        point = np.array([4,2])
        label = kd.predict_single(point)
    
    if source == 'multi':
        dataset = MultiClassDataset(n_samples=200, centers=4, n_features=2)
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        kd = KdTree(x, y, k=3).train()
        kd.save('./demo/')
        kd.load(path = '../demo/KdTree_k3_20190624_155846.pkl')
        
        point = np.array([10, 2])
        label = kd.predict_single(point)
        print('predict label is: %d'%label)
        
        kd.vis_boundary(plot_step=0.5)

