#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 09:04:13 2019

@author: ubuntu
"""

from core.multi_class_model_wrapper import OVOModel
from dataset.multi_class_dataset import MultiClassDataset
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    dataset = 'multi_class'
    
    if dataset == 'multi_class':
        dataset = MultiClassDataset(n_samples=300, centers=4, n_features=2)
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        ovo = OVOModel('logistic', train_x, train_y, lr=0.001, n_epoch=500, batch_size=64)
        acc1 = ovo.evaluation(train_x, train_y)
        acc2 = ovo.evaluation(test_x, test_y)
        print('train acc = %f, test acc= %f'%(acc1, acc2))
        
        ovo.vis_boundary()