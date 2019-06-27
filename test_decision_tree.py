#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""

import numpy as np
from dataset.loan_dataset import LoanDataset
from core.decision_tree_lib import CART, ID3, C45
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    source = 'loan'
    
    if source == 'treedata':  # from book of zhaozhiyong
        data = []
        with open('./dataset/simple/treedata.txt') as f:
            for line in f.readlines():
                sample = []
                lines = line.strip().split("\t")
                for x in lines:
                    sample.append(float(x))  # 转换成float格式
                data.append(sample)
        data = np.array(data)        # (200, 2)
        x = data[:, :-1]  # (400,2)
        y = data[:, -1]  # (400,)
        cart = CART(x, y, norm=False)
        cart.evaluation(x,y)
        cart.vis_boundary(plot_step=0.01)
    
    if source == 'loan': # from lihang
        dataset = LoanDataset()
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        cart = CART(x, y, norm=False)
        cart.evaluation(x, y)



