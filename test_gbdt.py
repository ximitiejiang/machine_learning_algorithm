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
from dataset.multi_class_dataset import MultiClassDataset
from core.random_forest_lib import RandomForest
from core.gbdt_lib import GBDT
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def label_transform(labels, label_transform_dict):
    """默认不改变label的取值范围，但可以通过该函数修改labels的对应范围
    例如svm需要label为[-1,1]，则可修改该函数。
    """
    new_labels = np.zeros(labels.shape)
    for i, label in enumerate(labels):
        new_label = label_transform_dict[label]
        new_labels[i] = int(new_label)   # 比如{1:1, 0:-1}就是要把1变为1, 0变为-1
    return new_labels
        
def label_to_one_hot(labels):
    """标签转换为独热编码：输入的labels需要是从0开始的整数，比如[0,1,2,...]
    输出的独热编码为[[1,0,0,...],
                  [0,1,0,...],
                  [0,0,1,...]]  分别代表0/1/2的独热编码
    """
    assert labels.ndim ==1, 'labels should be 1-dim array.'
    labels = labels.astype(np.int8)
    n_col = int(np.max(labels) + 1)   # 独热编码列数，这里可以额外增加列数，填0即可，默认是最少列数
    one_hot = np.zeros((labels.shape[0], n_col))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot  # (n_samples, n_col)

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
        # -1,1转0,1之后才能one hot
        y = label_transform(y, {-1:0, 1:1})
        # 独热编码
        y = label_to_one_hot(y)
        gb = GBDT(x, y,
                  n_clfs=20, learning_rate=0.5,
                  min_samples_split=3, max_depth=5,
                  min_impurity_reduction=1e-7).train()
        gb.evaluation(x,y)
        gb.vis_boundary(plot_step=0.01)
    

    if source == '5class':
        dataset = MultiClassDataset(n_samples=500, centers=4, n_features=2,
                                    center_box=(-8,+8), cluster_std=0.8,
                                    one_hot=True)
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        gb = GBDT(x, y,
                  n_clfs=2, learning_rate=0.5,
                  min_samples_split=3, max_depth=5,
                  min_impurity_reduction=1e-7).train()
        acc1 = gb.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = gb.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        gb.vis_boundary(plot_step=0.1)
        
    if source == 'moon':
        dataset = NonlinearDataset(type= 'moon', n_samples=500, noise=0.1, 
                                   one_hot=True)
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        # array to mat
#        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        gb = GBDT(x, y,
                  n_clfs=3, learning_rate=0.5,
                  min_samples_split=3, max_depth=5,
                  min_impurity_reduction=1e-7).train()
        acc1 = gb.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = gb.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        gb.vis_boundary(plot_step=0.05)

        
    if source == 'iris':
        dataset = IrisDataset(one_hot=True)
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        gb = GBDT(x, y,
                  n_clfs=20, learning_rate=0.5,
                  min_samples_split=3, max_depth=5,
                  min_impurity_reduction=1e-7).train()
        acc1 = gb.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        acc2 = gb.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        

        
        



