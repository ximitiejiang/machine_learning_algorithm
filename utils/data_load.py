#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:26:05 2019

@author: ubuntu
"""
import numpy as np

def batch_iterator(x, y, batch_size=64):
    """创建数据生成器，提供batch data"""
    n_samples = x.shape[0]
    
    for i in range(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        yield x[begin:end], y[begin:end]

        
class Dataloader():
    """创建一个简版迭代器"""
    def __init__(self, feats, labels, batch_size, shuffle=False):
        self.n_samples = feats.shape[0]
        self.batch_size = batch_size
        
        if shuffle:
            idx_shuffle = np.random.permutation(range(len(feats)))
            self.feats = feats[idx_shuffle]
            self.labels = labels[idx_shuffle]
        else:
            self.feats = feats
            self.labels = labels

        if batch_size == -1 or batch_size >= len(feats):
            self.begin = 0
            self.end = self.n_samples
        else:
            self.begin = 0
            self.end = self.begin + batch_size

    def __iter__(self):
        return self
    
    def __next__(self):
        self.begin = self.end
        self.end += self.batch_size
        if self.end < self.n_samples:
            return self.feats[self.begin - self.batch_size: self.end - self.batch_size], \
                    self.labels[self.begin - self.batch_size: self.end - self.batch_size]
        else:
            return self.feats[self.begin: self.n_samples], self.labels[self.begin: self.n_samples]
        
        