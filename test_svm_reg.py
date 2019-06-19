#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""

from dataset.heart_scale_dataset import HeartScaleDataset
from core.svm_reg_lib import SVMReg
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == "__main__":
    
    dataset = 'heart'
    
    if dataset == 'heart':
        filename = './dataset/simple/'  # 一个简单的2个特征的多分类数据集
        dataset = HeartScaleDataset(filename)
        
        x = dataset.datas    # (270, 13)
        y = dataset.labels   # (270,)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        svm = SVMReg(np.mat(train_x), np.mat(train_y).T)
        svm.train(alpha=0.5, n_epoch=10000, batch_size=64)
        acc = svm.cal_accuracy(train_x, train_y)
        
#        print('W = ', soft.W)
#        acc = soft.evaluation(test_x, test_y)
#        print('acc on test data is: %f'% acc)
#        
#        sample = np.array([2,8])
#        label, prob = soft.classify(sample)
#        print('one sample predict label = %d, probility = %f'% (label, prob))
    


        