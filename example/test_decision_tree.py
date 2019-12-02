#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""

import numpy as np
from dataset.loan_dataset import LoanDataset
from dataset.iris_dataset import IrisDataset
from dataset.nonlinear_dataset import NonlinearDataset
from core.decision_tree_lib import CARTClf, ID3Clf, C45Clf, CARTReg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    
    source = 'loan'
    
    if source == 'treedata':  # 2 classes: from book of zhaozhiyong
        data = []
        with open('../dataset/simple/treedata.txt') as f:
            for line in f.readlines():
                sample = []
                lines = line.strip().split("\t")
                for x in lines:
                    sample.append(float(x))  # 转换成float格式
                data.append(sample)
        data = np.array(data)        # (200, 2)
        idx = np.arange(len(data))
        idx = np.random.permutation(idx)
        data = data[idx]
        
        x = data[:, :-1]  # (400,2)
        y = data[:, -1]  # (400,)
        cart = CARTClf(x, y).train()
        cart.evaluation(x,y)
        cart.vis_boundary(plot_step=0.01)
    
    if source == 'loan': # from lihang
        dataset = LoanDataset()
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        id3 = ID3Clf(x, y).train()
        acc1 = id3.evaluation(x, y)
        
        cart = CARTClf(x, y).train()
        acc2 = cart.evaluation(x, y)
        print("final tree depth: %d, final gini: %d"%(cart.tree_final_params['final_depth'],
                                                      cart.tree_final_params['final_gini']))
    
    if source == 'moon':
        dataset = NonlinearDataset(type= 'circle', n_samples=300, noise=0.05, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        # array to mat
#        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        # 对比cart
        cart = CARTClf(train_x, train_y, min_samples_split=2).train()
        acc1 = cart.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = cart.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        cart.vis_boundary(plot_step=0.05)
        
        # 对比id3
        id3 = ID3Clf(train_x, train_y, min_samples_split=2).train()
        acc1 = id3.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = id3.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        id3.vis_boundary(plot_step=0.05)
        
        # 对比c45
        c45 = C45Clf(train_x, train_y, min_samples_split=2).train()
        acc1 = c45.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = c45.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        c45.vis_boundary(plot_step=0.05)

        
    if source == 'iris':
        dataset = IrisDataset()
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

        # 对比cart
        cart = CARTClf(train_x, train_y, min_samples_split=2).train()
        acc1 = cart.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = cart.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        # 对比id3
        id3 = ID3Clf(train_x, train_y, min_samples_split=2).train()
        acc1 = id3.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = id3.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        # 对比c45
        c45 = C45Clf(train_x, train_y, min_samples_split=2).train()
        acc1 = c45.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = c45.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        





