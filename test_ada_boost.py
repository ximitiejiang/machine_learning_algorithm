#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""

import numpy as np
from dataset.loan_dataset import LoanDataset
from dataset.iris_dataset import IrisDataset
from dataset.breast_cancer_dataset import BreastCancerDataset
from dataset.nonlinear_dataset import NonlinearDataset
from dataset.multi_class_dataset import MultiClassDataset
from core.ada_boost_lib import AdaBoost
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    source = 'moon'
    
    if source == 'treedata':  # 2 classes: from book of zhaozhiyong
        data = []
        with open('./dataset/simple/treedata.txt') as f:
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
        ab = AdaBoost(x, y, n_clfs=10).train()
        ab.evaluation(x,y)
        ab.vis_boundary(plot_step=0.01)
    
    if source == 'moon':
        dataset = NonlinearDataset(type= 'circle', n_samples=500, noise=0.05, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        # array to mat
#        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        ab = AdaBoost(x, y, n_clfs=200).train()
        acc1 = ab.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = ab.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        ab.vis_boundary(plot_step=0.05)

    if source == 'cancer':
        dataset = BreastCancerDataset(label_transform_dict={1:1,0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        ab = AdaBoost(x, y, n_clfs=10).train()

        acc = ab.evaluation(train_x, train_y)
        print('training acc = %f'%(acc))
        
        acc2 = ab.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))   

    if source == '2class2':  # TODO: loss curve still need optimize
        import pandas as pd       
        filename = './dataset/simple/2classes_data_2.txt'  # 一个简单的2个特征的2分类数据集
        data = pd.read_csv(filename, sep='\t').values
        x = data[:,0:2]
        y = data[:,-1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        
        ab = AdaBoost(x, y, n_clfs=10).train()
        acc = ab.evaluation(train_x, train_y)
        print('training acc = %f'%(acc))
        
        acc2 = ab.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))    
        
        ab.vis_boundary()
        



