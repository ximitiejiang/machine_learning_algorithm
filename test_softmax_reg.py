#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""

from dataset.mnist_dataset import MnistDataset
from dataset.digits_dataset import DigitsDataset

from core.softmax_reg_lib import SoftmaxReg
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    dataset = 'digits'
    
    if dataset == 'mnist':        # acc = 0.843 但必须特征归一化， 同时w必须初始化为0，否则会导致inf问题
        dataset = MnistDataset(data_type='train')  # 采用mnist数据集
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        soft = SoftmaxReg(train_feats, train_labels)
        soft.train(alpha=0.0001, n_epoch=10, batch_size=32)
        
        # evaluation
        acc = soft.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    
    if dataset == 'digits':       # acc = 0.966
        dataset = DigitsDataset(data_type='train')
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        soft = SoftmaxReg(train_feats, train_labels)
        soft.train(alpha=0.0001, n_epoch=1000, batch_size=16)
        
        # evaluation
        acc = soft.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    

        