#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""

from dataset.heart_scale_dataset import HeartScaleDataset
from dataset.nonlinear_dataset import NonlinearDataset
from core.svm_lib import SVMC
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == "__main__":
    
    dataset = 'circle'
    
    if dataset == 'heart':
        filename = './dataset/simple/'  # 一个简单的2个特征的多分类数据集
        dataset = HeartScaleDataset(filename)
        
        x = dataset.datas    # (270, 13)
        y = dataset.labels   # (270,)   取值1, -1
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2) # (n, 13) (n,)
        # array to mat
        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        svm = SVMC(train_x, train_y.T, 
                     C=5, toler=0.001, max_iter=500, 
                     kernel_option=('rbf', 0.9))
        svm.train()
        acc = svm.cal_accuracy(train_x, train_y.T)
        print('training acc = %f'%(acc))
        
        acc2 = svm.cal_accuracy(test_x, test_y.T)
        print('test acc = %f'%(acc2))
        

    if dataset == 'moon':
        # acc=0.98@C=5, sigma=0.45
        dataset = NonlinearDataset(type= 'moon', n_samples=300, noise=0.1, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        # array to mat
        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        svm = SVMC(train_x, train_y.T, 
                     C=1, toler=0.001, max_iter=500, 
                     kernel_option=('rbf', 0.45))
        svm.train()
        acc = svm.cal_accuracy(train_x, train_y.T)
        print('training acc = %f'%(acc))
        
        acc2 = svm.cal_accuracy(test_x, test_y.T)
        print('test acc = %f'%(acc2))
        
        svm.vis_boundary(plot_step=0.05)
        
    if dataset == 'circle':
        # acc=0.98@C=5, sigma=0.5
        dataset = NonlinearDataset(type= 'circle', n_samples=300, noise=0.05, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        # array to mat
        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        svm = SVMC(train_x, train_y.T, 
                     C=10, toler=0.001, max_iter=500, 
                     kernel_option=('rbf', 0.01))
        svm.train()
        acc = svm.cal_accuracy(train_x, train_y.T)
        print('training acc = %f'%(acc))
        
        acc2 = svm.cal_accuracy(test_x, test_y.T)
        print('test acc = %f'%(acc2))
        
        svm.vis_boundary(plot_step=0.05)
        svm.save(path = './demo')

        
    if dataset == 'load':
        # acc=0.98@C=5, sigma=0.5
        dataset = NonlinearDataset(type= 'circle', n_samples=300, noise=0.02, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        # array to mat
        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        svm = SVMC(train_x, train_y.T, 
                   C=5, toler=0.001, max_iter=500, 
                   kernel_option=('rbf', 0.5))
        
        # 不进行训练，直接加载参数进行预测
        svm.load(path = './demo/SVMC_20190621_232511.pkl')
        acc = svm.cal_accuracy(test_x, test_y.T)
        print('test acc = %f'%(acc))
        svm.vis_boundary(plot_step=0.05)

        