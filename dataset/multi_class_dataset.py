#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

class MultiClassDataset():
    """sklearn自带多分类数据集，可生成2-n任意类别数据集，基本是线性可分，但有可能部分点线性不可分
    
    """
    def __init__(self, n_samples=100, centers=2, n_features=2):
        self.datas, self.labels = make_blobs(n_samples=n_samples, 
                                             centers=centers, 
                                             n_features=n_features)

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]
        
        return [data, label]
    
    def statistics(self):
        classes = set(self.labels)
        n_classes = len(classes)
        
        class_num_dict = {}
        for label in self.labels:
            class_num_dict[label] = class_num_dict.get(label, 0) + 1
        print('num_classes: %d'%n_classes)
        for key, value in sorted(class_num_dict.items()):
            print('class %d: %d' % (key, value))
        
        # show scatter points
        if self.datas.shape[1] == 2:
            color = [c*64 + 128 for c in self.labels.reshape(-1)]
            plt.scatter(self.datas[:,0], self.datas[:,1], c=color)
    

if __name__ == '__main__':
    mc = MultiClassDataset(100, 4, 2)
    data, label = mc[10]                     
    
    mc.statistics()                 