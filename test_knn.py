#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""


import matplotlib.pyplot as plt
from dataset.digits_dataset import DigitsDataset
from dataset.multi_class_dataset import MultiClassDataset
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
    
    if source == 'multi':
        dataset = MultiClassDataset(n_samples=100, centers=5, n_features=2)
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        knn = KNN(dataset.datas, dataset.labels, k=5)
        knn.vis_boundary()
        
    if source = 'compare'
        dataset = MultiClassDataset(n_samples=100, centers=5, n_features=2)
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        kdt = KdTree(dataset.datas, dataset.labels, k=3)
        kdt.vis_boundary()