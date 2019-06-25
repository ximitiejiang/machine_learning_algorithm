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
from core.naive_bayes_lib import NaiveBayes 

if __name__ == "__main__":
    
    source = 'multi'
    
    if source == 'multi':
        dataset = MultiClassDataset(n_samples=500, centers=4, n_features=2)
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        nb = NaiveBayes(dataset.datas, dataset.labels, k=5)
        nb.vis_boundary()
    
    if source == 'circle':
        # acc=0.98@C=5, sigma=0.5
        dataset = NonlinearDataset(type= 'circle', n_samples=300, noise=0.05, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        # array to mat
        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        nb = NaiveBayes(dataset.datas, dataset.labels, k=5)

        acc = nb.evaluation(train_x, train_y.T)
        
        nb.vis_boundary(plot_step=0.05)
    
    if source == 'digits':
        # get dataset
        dataset = DigitsDataset(data_type = 'train')
        # get model
        nb = NaiveBayes(dataset.datas, dataset.labels)
        # get sample
        sample_id = 1507
        sample, label = dataset[sample_id]  # 用第2000个样本做测试
        # test and show
        pred = nb.predict_single(sample)
        print("the sample label is %d, predict is %d"%(label, pred))   
        plt.figure()
        plt.subplot(1,1,1)
        plt.imshow(sample.reshape(8,8), cmap='gray')
