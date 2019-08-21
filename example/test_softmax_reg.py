#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""

from dataset.mnist_dataset import MnistDataset
from dataset.digits_dataset import DigitsDataset
from dataset.multi_class_dataset import MultiClassDataset

from core.softmax_reg_lib import SoftmaxReg
from sklearn.model_selection import train_test_split
import numpy as np
from utils.transformer import label_to_onehot

if __name__ == "__main__":
    
    dataset = 'digits'
    
    if dataset == '4class':
        import pandas as pd
        filename = '../dataset/simple/4classes_data.txt'  # 一个简单的2个特征的多分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        train_y = label_to_onehot(train_y)
        test_y = label_to_onehot(test_y)
        
        soft = SoftmaxReg(train_x, train_y, lr=0.1, n_epoch=6000, batch_size=-1)
        soft.train()  # 在学习率0.5下精度在0.8-0.9之间，太小学习率导致精度下降
        print('W = ', soft.W)
        acc = soft.evaluation(test_x, test_y)
        print('acc on test data is: %f'% acc)
        
        soft.vis_boundary(plot_step=0.1)
        
    
    if dataset == 'multi':
        dataset = MultiClassDataset(n_samples=500, centers=4, n_features=2, cluster_std=0.5,
                                    one_hot=True)
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        soft = SoftmaxReg(train_feats, train_labels, lr=0.01, n_epoch=600, batch_size=64)
        soft.train()
        # evaluation
        acc = soft.evaluation(test_feats, test_labels)
        print('test acc = %f'%acc)
        
        soft.vis_boundary(plot_step=0.1)
        
    if dataset == 'mnist':        # 必须特征归一化，同时w必须初始化为0，同事batch_size不能太大，当前设为64
        # acc = 0.843@lr0.0001/batch32/w0/norm
        dataset = MnistDataset(data_type='train', 
                               norm=True, one_hot=True)  # 采用mnist数据集
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        soft = SoftmaxReg(train_feats, train_labels, lr=0.01, n_epoch=100, batch_size=64)
        soft.train()
        
        # evaluation
        acc = soft.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    
    if dataset == 'digits':       
        # acc = 0.966@lr0.0001/batch16/w0/norm
        dataset = DigitsDataset(norm=True, one_hot=True)
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        soft = SoftmaxReg(train_feats, train_labels, lr=0.01, n_epoch=1000, batch_size=128)
        soft.train()
        
        # evaluation
        acc = soft.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    

        