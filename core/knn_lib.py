#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:59:16 2019

@author: ubuntu
"""
import numpy as np


class KNNLib():
    def __init__(self, feats, labels):
        """ knn algorithm lib
        Args:
            feats(numpy): (n,2)
            labels(numpy): (n,)
            k(int): 
        Returns:
            
        """
        self.feats = feats
        self.labels = labels
    
    def classify(self, data, k):
        """
        Args:
            data(numpy): (1,2)
            k(int): 
        Returns:
            
        """
        assert(data, np.ndarray), 'data should be ndarray.'
        # calculate distance
        tiled_data = np.tile(data, (self.feats.shape(0),1))           # 把输入数据堆叠成特征的高度
        dist = np.sqrt(np.sum((tiled_data - self.feats)**2, axis=1))  # distances = np.sqrt((xi - x)^2) 
        
        # sort distance and vote
        dist_sort_index = np.argsort(dist)
        for i in range(k):
            vote = data[dist_sort_index[i]]
            
        return vote

        
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

class DigitsDataset():
    """sklearn自带手写数字集(1797,64)共计1797个样本
    digits.keys() = ['images','data','target_names','DESCR','target']
    其中imgaes为(8,8)ndarray, data为(1797,64)ndarray展开的图片值，
    
    """
    def __init__(self):
        self.dataset = load_digits()        # dict, len=5

    def __len__(self):
        return len(self.dataset['images'])
    
    def __getitem__(self, idx):
        img = self.dataset.images[idx]
        label = self.dataset.target[idx]
        
        return [img, label] 

if __name__ == "__main__":
    data = DigitsDataset()

    img1, label1 = data[0]
    img2, label2 = data[1]
    img3, label3 = data[2]    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.subplot(1,3,2)
    plt.imshow(img2)    
    plt.subplot(1,3,3)
    plt.imshow(img3)    
#    knn = KNNLib(feats, labels)
        