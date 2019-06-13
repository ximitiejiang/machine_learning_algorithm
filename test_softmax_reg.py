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
from core.softmax_reg_lib import SoftmaxReg
from sklearn.model_selection import train_test_split

def accuracy(preds, labels):
    """preds(n,), labels(n)
    """
    return acc

if __name__ == "__main__":
    
    dataset = 'digits'
    
    if dataset == 'mnist':        # acc = 0.98
        dataset = MnistDataset(data_type='train')  # 采用mnist数据集
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        soft = SoftmaxReg(train_feats, train_labels)
        soft.train(n_epoch=1000)
        
        # evaluation
        acc = soft.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    
    if dataset == 'digits':       # acc = 1
        dataset = DigitsDataset(data_type='train')
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        soft = SoftmaxReg(train_feats, train_labels)
        soft.train(n_epoch=10000)
        
        # evaluation
        acc = soft.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    

        