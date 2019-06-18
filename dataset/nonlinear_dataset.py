#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_circles
import matplotlib.pyplot as plt

class NonlinearDataset():
    """sklearn自带非线性数据集，可生成二分类的非线性可分数据集
    有两种数据集可选，一种moon月亮形，一种circle圆圈形
    """
    def __init__(self, type= 'moon', n_samples=100, noise=0.1):
        if type == 'moon':
            self.datas, self.labels = make_moons(n_samples=n_samples, 
                                                 noise=noise)
        elif type == 'circle':
            self.datas, self.labels = make_circles(n_samples=n_samples, 
                                                   noise=noise)
        else:
            print('wrong type input.')
            
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
    mc = NonlinearDataset('circle', 100, 0.05)
    data, label = mc[10]                     
    
    mc.statistics()                 