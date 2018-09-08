#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 21:39:34 2018

@author: suliang
"""

# PCA主要用来给特征降维，本function主要用sklearn的现成函数

def loadDataset():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=10000, 
                      n_features=3, 
                      centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], 
                      cluster_std=[0.2, 0.1, 0.2, 0.2],
                      random_state =9)
    return X, y


def std_PCA():
    pass



if __name__ == '__main__':
    
    test_id = 0
    
    if test_id == 0:
        data, label = loadDataset()
        
        plt.scatter(data[:,0],data[:,1])