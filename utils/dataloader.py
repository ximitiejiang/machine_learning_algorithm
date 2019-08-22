#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:26:05 2019

@author: ubuntu
"""
import numpy as np

def train_test_split(X, y, test_size=0.3, shuffle=True, seed=None):
    """ 分割数据集 """
    if shuffle:
        if seed:
            np.random.seed(seed)
        idx = np.arange(X.shape[0])
        idx = np.random.permutation(idx)
        X, y = X[idx], y[idx]
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def batch_iterator(X, y, batch_size=-1):
    """最简版函数形式的dataloader"""   
    if batch_size == -1:
        yield X, y
    else:
        n_samples = X.shape[0]
        for i in np.arange(0, n_samples, batch_size):
            begin, end = i, min(i+batch_size, n_samples)  # 确保最后一部分也能被使用
            yield X[begin:end], y[begin:end]

        
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
        
        