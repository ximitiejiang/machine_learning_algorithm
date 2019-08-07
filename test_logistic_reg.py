#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""
from dataset.breast_cancer_dataset import BreastCancerDataset
from dataset.mnist_dataset import MnistDataset
from dataset.digits_dataset import DigitsDataset
from core.logistic_reg_lib import LogisticReg
from core.svm_lib import SVMC
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    
    dataset = '2class2'
    
    if dataset == '2class':  # TODO: loss curve still need optimize
        import pandas as pd       
        filename = './dataset/simple/2classes_data.txt'  # 一个简单的2个特征的2分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        logs = LogisticReg(train_x, train_y, lr=0.001, n_epoch=1000, batch_size=64)
        logs.train()
        print('W = ', logs.W)
        acc = logs.evaluation(test_x, test_y)
        print('acc on test data is: %f'% acc)
        
        logs.vis_boundary()
        
        # compare with svm
        # array to mat
        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        for i, label in enumerate(train_y.tolist()[0]):
            if label == 0:
                train_y[0, i] = -1
        for i, label in enumerate(test_y.tolist()[0]):
            if label == 0:
                test_y[0, i] = -1
                
        svm = SVMC(train_x, train_y.T, 
                     C=1, toler=0.001, max_iter=500, 
                     kernel_option={'type':'linear'})  # 采用线性核
        svm.train()
        acc = svm.evaluation(train_x, train_y.T)
        print('svm training acc = %f'%(acc))
        
        acc2 = svm.evaluation(test_x, test_y.T)
        print('svm test acc = %f'%(acc2))
        
        svm.vis_boundary(plot_step=0.05)
        
    if dataset == '2class2':  # TODO: loss curve still need optimize
        import pandas as pd       
        filename = './dataset/simple/2classes_data_2.txt'  # 一个简单的2个特征的2分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        logs = LogisticReg(train_x, train_y, lr=0.001, n_epoch=1000, batch_size=64)
        logs.train()
        print('W = ', logs.W)
        acc = logs.evaluation(test_x, test_y)
        print('acc on test data is: %f'% acc)    
        
        logs.vis_boundary()
        
        # compare with svm
        # array to mat
        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        for i, label in enumerate(train_y.tolist()[0]):
            if label == 0:
                train_y[0, i] = -1
        for i, label in enumerate(test_y.tolist()[0]):
            if label == 0:
                test_y[0, i] = -1
                
        svm = SVMC(train_x, train_y.T, 
                     C=1, toler=0.001, max_iter=500, 
                     kernel_option={'type':'linear'})  # 采用线性核
        svm.train()
        acc = svm.evaluation(train_x, train_y.T)
        print('svm training acc = %f'%(acc))
        
        acc2 = svm.evaluation(test_x, test_y.T)
        print('svm test acc = %f'%(acc2))
        
        svm.vis_boundary(plot_step=0.05)
        
    if dataset == 'mnist':      
        # acc = 0.98@lr0.001/batch128/w0/norm
        dataset = MnistDataset(data_type='train', binary=True)  # 采用mnist数据集
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        logi = LogisticReg(train_feats, train_labels, lr=0.001, n_epoch=100, batch_size=128)
        logi.train()
        
        # evaluation
        acc = logi.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    
    if dataset == 'digits':       
        # acc = 0.996@lr0.001/batch128/w0/norm
        dataset = DigitsDataset(data_type='binary')
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        logi = LogisticReg(train_feats, train_labels, lr=0.001, n_epoch=500, batch_size=128)
        logi.train()
        
        # evaluation
        acc = logi.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    
    if dataset == 'breastcancer':    
        # acc = 0.94@lr0.007/batch128/w0/norm
        dataset = BreastCancerDataset()
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        logi = LogisticReg(train_feats, train_labels, lr=0.007, n_epoch=100, batch_size=128)
        logi.train()
        
        # evaluation
        acc = logi.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
        