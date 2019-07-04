#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

class RegressionDataset():
    """sklearn自带回归数据集，
    """
    def __init__(self, n_samples=100, n_features=2, noise=1):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
#        super().__init__()
        self.dataset = self.get_dataset()
        self.datas = self.dataset[0]
        self.labels = self.dataset[1]
        
    
    def get_dataset(self):
        return make_regression(n_samples=self.n_samples,
                               n_features=self.n_features,
                               noise=self.noise)
    
if __name__ == "__main__":
    reg = RegressionDataset()
    plt.scatter(reg.datas[:,0], reg.datas[:,1])