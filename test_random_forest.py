#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""

import numpy as np
from dataset.loan_dataset import LoanDataset
from dataset.iris_dataset import IrisDataset
from dataset.nonlinear_dataset import NonlinearDataset
from core.random_forest_lib import RandomForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    source = 'treedata'
    
    if source == 'treedata':  # 2 classes: from book of zhaozhiyong
        data = []
        with open('./dataset/simple/treedata.txt') as f:
            for line in f.readlines():
                sample = []
                lines = line.strip().split("\t")
                for x in lines:
                    sample.append(float(x))  # 转换成float格式
                data.append(sample)
        data = np.array(data)        # (200, 2)
        idx = np.arange(len(data))
        idx = np.random.permutation(idx)
        data = data[idx]
        
        x = data[:, :-1]  # (400,2)
        y = data[:, -1]  # (400,)
        rf = RandomForest(x, y, n_trees=100, sub_samples_ratio=0.5)
        rf.evaluation(x,y)
        rf.vis_boundary(plot_step=0.01)
    
    if source == 'loan': # from lihang
        dataset = LoanDataset()
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        rf = RandomForest(x, y, n_trees=30, sub_samples_ratio=0.5)
        rf.evaluation(x, y)

    
    if source == 'moon':
        dataset = NonlinearDataset(type= 'moon', n_samples=300, noise=0.05, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        # array to mat
#        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        rf = RandomForest(train_x, train_y, n_trees=100, sub_samples_ratio=0.5)
        acc1 = rf.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = rf.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        rf.vis_boundary(plot_step=0.05)

        
    if source == 'iris':
        dataset = IrisDataset()
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        rf = RandomForest(train_x, train_y, n_trees=100, sub_samples_ratio=0.5)
        acc1 = rf.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        acc2 = rf.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        

        
        



