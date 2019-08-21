#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""


import matplotlib.pyplot as plt
import numpy as np
from dataset.digits_dataset import DigitsDataset
from dataset.multi_class_dataset import MultiClassDataset
from dataset.nonlinear_dataset import NonlinearDataset
from sklearn.model_selection import train_test_split
from core.knn_lib import KNN
from core.kd_tree_lib import KdTree

if __name__ == "__main__":
    
    source = 'multi'
    
    if source == 'digits':
        # get dataset
        dataset = DigitsDataset(data_type = 'train')
        # get model
        knn = KNN(dataset.datas, dataset.labels, k=5)
        # get sample
        sample_id = 1507
        sample, label = dataset[sample_id]  # 用第2000个样本做测试
        # test and show
        pred = knn.predict_single(sample)
        print("the sample label is %d, predict is %d"%(label, pred))   
        plt.figure()
        plt.subplot(1,1,1)
        plt.imshow(sample.reshape(8,8), cmap='gray')
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        knn.evaluation(test_x, test_y)
    
    if source == 'multi':
        dataset = MultiClassDataset(n_samples=500, centers=4, n_features=2)
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        knn = KNN(dataset.datas, dataset.labels, k=5)
        knn.vis_boundary(plot_step=0.5)
    
    if source == 'circle':
        dataset = NonlinearDataset(type= 'circle', n_samples=300, noise=0.05, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        
        knn = KNN(dataset.datas, dataset.labels, k=5)
        acc = knn.evaluation(train_x, train_y.T)        
        knn.vis_boundary(plot_step=0.05)
        
    if source == 'moon':
        dataset = NonlinearDataset(type= 'moon', n_samples=300, noise=0.05, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        
        knn = KNN(dataset.datas, dataset.labels, k=5)
        acc = knn.evaluation(train_x, train_y.T)        
        knn.vis_boundary(plot_step=0.05)
    
    if source == 'compare':
        dataset = MultiClassDataset(n_samples=100, centers=5, n_features=2)
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        kdt = KdTree(dataset.datas, dataset.labels, k=3)
        kdt.vis_boundary()