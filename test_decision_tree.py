#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""

import numpy as np
from dataset.multi_class_dataset import MultiClassDataset
from core.decision_tree_lib import CART, ID3, C45
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    source = 'sine'
    
    if source == 'sine':  # points from Lihang's book(P42)
        data = []
        with open('./dataset/simple/sine.txt') as f:
            for line in f.readlines():
                sample = []
                lines = line.strip().split("\t")
                for x in lines:
                    sample.append(float(x))  # 转换成float格式
                data.append(sample)
        data = np.array(data)     # (200, 2)
        x = data[:,0]
        y = data[:,1]
        plt.scatter(x,y)
        cart = CART(x,y,30,0.3)
        cart.cal_error()
        point = np.array([4,2])
        label = cart.predict_single(point)
    
    if source == 'multi':
        dataset = MultiClassDataset(n_samples=200, centers=4, n_features=2)
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        cart = CART(x, y)
        cart.evaluation(train_x, train_y)
        cart.evaluation(test_x, test_y)
#        cart.save('./demo/')
#        cart.load(path = './demo/KdTree_k3_20190624_155846.pkl')
#        
#        point = np.array([10, 2])
#        label = kd.predict_single(point)
#        print('predict label is: %d'%label)
        cart.vis_boundary(plot_step=0.1)

