#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""
from dataset.breast_cancer_dataset import BreastCancerDataset
from dataset.mnist_dataset import MnistDataset
from dataset.digits_dataset import DigitsDataset
from core.logistic_reg_lib import LogisticReg, LogisticReg_autograd, LogisticReg_pytorch
from sklearn.model_selection import train_test_split
import numpy as np
from utils.transformer import label_transform
import pandas as pd 

if __name__ == "__main__":
    
    dataset = 'breastcancer'
    
    if dataset == '2class':      
        filename = '../dataset/simple/2classes_data.txt'  # 一个简单的2个特征的2分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        logs = LogisticReg(train_x, train_y, lr=0.001, n_epoch=2000, batch_size=-1)
        logs.train()
        print('W = ', logs.W)
        acc = logs.evaluation(test_x, test_y)
        print('acc on test data is: %f'% acc)
        
        logs.vis_boundary()
        
        
    if dataset == '2class2':     
        filename = '../dataset/simple/2classes_data_2.txt'  # 一个简单的2个特征的2分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        
        # 由于标签类型为-1,1, 且没有数据集类，所以需要手动需要转换为0,1
        y = label_transform(y, label_transform_dict={1:1, -1:0, 0:0})
        
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        logs = LogisticReg(train_x, train_y, lr=0.001, n_epoch=1000, batch_size=-1)
        logs.train()
        print('W = ', logs.W)
        acc = logs.evaluation(test_x, test_y)
        print('acc on test data is: %f'% acc)    
        
        logs.vis_boundary()
        
        
    if dataset == 'mnist':    # TODO: digits增加normalize后就可以跑通，但mnist还是会梯度爆炸,需要减小batchsize或减小学习率
        
        dataset = MnistDataset(data_type='train', binary=True, norm=True)  # 采用mnist数据集
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        logi = LogisticReg(train_feats, train_labels, lr=0.0001, n_epoch=100, batch_size=32)
        logi.train()
        
        logi.evaluation(train_feats, train_labels)  
        # evaluation
        logi.evaluation(test_feats, test_labels)
    
    if dataset == 'digits':       
        # 注意，对于图片等数据，至少需要norm之后再训练，否则loss会爆炸
        dataset = DigitsDataset(binary=True, norm=True)
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        logi = LogisticReg(train_feats, train_labels, lr=0.001, n_epoch=500, batch_size=-1)
        logi.train()
        
        # evaluation
        acc = logi.evaluation(test_feats, test_labels)
    
    if dataset == 'breastcancer':    
        # 必须增加norm，否则梯度爆炸
        dataset = BreastCancerDataset(norm=True)
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        logi = LogisticReg(train_feats, train_labels, lr=0.007, n_epoch=100, batch_size=128)
        logi.train()
        
        # evaluation
        acc = logi.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    
    
    if dataset == "2class_autograd":
        filename = './dataset/simple/2classes_data.txt'  # 一个简单的2个特征的2分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        logs = LogisticReg_autograd(train_x, train_y, lr=0.001, n_epoch=2000, batch_size=-1)
        logs.train()
        print('W = ', logs.weights)
#        logs.evaluation(test_x, test_y)
#        
#        logs.vis_boundary()
        
    if dataset == "2class_pytorch":
        filename = './dataset/simple/2classes_data.txt'  # 一个简单的2个特征的2分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        logs = LogisticReg_pytorch(train_x, train_y, lr=0.001, n_epoch=2000, batch_size=-1)
#        loss_fn
#        optimizer
        logs.train()
        print('W = ', logs.weights)