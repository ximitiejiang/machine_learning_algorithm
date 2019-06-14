#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""

import matplotlib.pyplot as plt
from dataset.breast_cancer_dataset import BreastCancerDataset
from dataset.mnist_dataset import MnistDataset
from dataset.digits_dataset import DigitsDataset
from core.logistic_reg_lib import LogisticReg
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    dataset = 'breastcancer'
    
    if dataset == 'mnist':        # acc = 0.98
        dataset = MnistDataset(data_type='train_binary')  # 采用mnist数据集
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        logi = LogisticReg(train_feats, train_labels)
        logi.train(alpha=0.001, n_epoch=100, batch_size=128)
        
        # evaluation
        acc = logi.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    
    if dataset == 'digits':       # acc = 0.996
        dataset = DigitsDataset(data_type='binary')
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        logi = LogisticReg(train_feats, train_labels)
        logi.train(alpha=0.001, n_epoch=500, batch_size=128)
        
        # evaluation
        acc = logi.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    
    if dataset == 'breastcancer':    # 原有acc = 0.56, 增加特征归一化以后acc变为0.94, 学习率0.001收敛，但设置小了反而acc下降
        dataset = BreastCancerDataset()
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        logi = LogisticReg(train_feats, train_labels)
        logi.train(alpha=0.007, n_epoch=100, batch_size=128)
        
        # evaluation
        acc = logi.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
        