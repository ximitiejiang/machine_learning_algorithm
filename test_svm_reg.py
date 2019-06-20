#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""

from dataset.heart_scale_dataset import HeartScaleDataset
from dataset.nonlinear_dataset import NonlinearDataset
from core.svm_reg_lib import SVMReg
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == "__main__":
    
    dataset = 'heart'
    
    if dataset == 'heart':
        filename = './dataset/simple/'  # 一个简单的2个特征的多分类数据集
        dataset = HeartScaleDataset(filename)
        
        x = dataset.datas    # (270, 13)
        y = dataset.labels   # (270,)   取值1, -1
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2) # (n, 13) (n,)
        
        svm = SVMReg(np.mat(train_x), np.mat(train_y).T, 
                     C=5, toler=0.001, max_iter=500, 
                     kernel_option=('rbf', 0.45))
        svm.train()
        acc = svm.cal_accuracy(train_x, train_y)
        print('training acc = %f'%(acc))
        
        acc2 = svm.cal_accuracy(test_x, test_y)
        print('test acc = %f'%(acc2))
        

    if dataset == 'nonlinear':
        
        dataset = NonlinearDataset(type= 'moon', n_samples=300, noise=0.1)
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        
        svm = SVMReg(np.mat(train_x), np.mat(train_y).T, 
                     C=5, toler=0.001, max_iter=500, 
                     kernel_option=('rbf', 0.45))
        svm.train()
        acc = svm.cal_accuracy(train_x, train_y)
        print('training acc = %f'%(acc))
        
        acc2 = svm.cal_accuracy(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        

        