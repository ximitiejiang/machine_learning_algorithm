#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""

from dataset.mnist_dataset import MnistDataset
from dataset.digits_dataset import DigitsDataset

from core.perceptron_lib import Perceptron
from sklearn.model_selection import train_test_split
import pandas as pd


if __name__ == "__main__":
    
    dataset = '2class'
    
    if dataset == '2class':
        # acc = 0.90@lr0.01/batch1/w0/norm
        filename = './dataset/simple/2classes_data.txt'  # 一个简单的2个特征的2分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        # 变换标签[1,0]->[1,-1]
        for i, label in enumerate(y):
            if label == 0:
                y[i] = -1
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        perc = Perceptron(train_x, train_y)
        perc.train(lr=0.01, n_epoch=100, batch_size=1)
        print('W = ', perc.W)
        acc = perc.evaluation(test_x, test_y)
        print('acc on test data is: %f'% acc)
        
        perc.vis_boundary()
    
    if dataset == '2class2':
        # acc = 0.95@lr0.01/batch1/w0/norm
        filename = './dataset/simple/2classes_data_2.txt'  # 一个简单的2个特征的2分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        # 变换标签[1,0]->[1,-1]
        for i, label in enumerate(y):
            if label == 0:
                y[i] = -1
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        perc = Perceptron(train_x, train_y)
        perc.train(lr=0.01, n_epoch=1000, batch_size=1)
        print('W = ', perc.W)
        acc = perc.evaluation(test_x, test_y)
        print('acc on test data is: %f'% acc)
        
        perc.vis_boundary()
    
    if dataset == 'mnist':        # 必须特征归一化，同时w必须初始化为0，否则会导致inf问题
        # acc = 0.985@lr0.00001/batch8/w0/norm
        dataset = MnistDataset(data_type='train_binary')  # 采用mnist数据集
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        soft = Perceptron(train_feats, train_labels)
        soft.train(lr=0.00001, n_epoch=10, batch_size=8)
        
        # evaluation
        acc = soft.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    
    if dataset == 'digits':       
        # acc = 0.994@lr0.0001/batch64/w0/norm
        dataset = DigitsDataset(data_type='binary')
        train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.3)
        
        # get model
        soft = Perceptron(train_feats, train_labels)
        soft.train(lr=0.0001, n_epoch=1000, batch_size=64)
        
        # evaluation
        acc = soft.evaluation(test_feats, test_labels)
        print('acc = %f'%acc)
    

        