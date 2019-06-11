#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""

import numpy as np

class LogisticReg():
    def __init__(self, feats, labels):
        """ logistic reg algorithm lib, 当前只支持2分类
        logistic reg算法由一个线性模块(w0x0+w1x1+..wnxn)和一个非线性模块(sigmoid函数)组成一个函数
        用输入特征feats和labels来训练这个模块，得到一组(w0,w1,..wn)的模型，可用来进行二分类问题的预测，但不能直接用于多分类问题
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)
        """
        assert feats.ndim ==2, 'the feats should be (n_samples, m_feats), each sample should be 1-dim flatten data.'
        self.feats = feats
        self.labels = labels
        self.trained = False
    
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))
            
    def train(self, alpha=0.001, n_epoch=500):
        """feats(x1,x2,..xn) -> feats(1,x1,x2,..xn)
        Args:
            alpha(float): 梯度下降步长
            n_epoch(inf): 循环训练轮数
        """
        
        feats = np.concatenate([np.ones((len(self.feats),1)), self.feats], axis=1)  # (n_sample, 1+ n_feats) (1,x1,x2,..xn)
        labels = self.labels.reshape(-1, 1)   # (n_sample, 1)
        self.W = np.ones((feats.shape[1], 1)) 
        
        for i in range(n_epoch):
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn, 然后通过sigmoid函数转换为概率(0-1)
            # (n_sample, 1) 每个样本一个prob(0~1)，也就是作为2分类问题的预测概率
            hw = np.dot(feats, self.W)
            prob = self.sigmoid(hw)  # (455,31) dot (31,1)
            gradient = np.dot(feats.transpose(), (prob - labels))  # grad = (y-y')*x  (3,n)dot((n,1)-(n,1))->(3,n)dot(n,1)->(3,)
            self.W -= alpha * gradient   #(m,1) 
        self.trained = True
        
    def classify(self, feats):
        """
        Args:
            data(numpy): (1, m_feats) or (m_feats,)，如果是(m,n)则需要展平
            k(int): number of neibours
        Returns:
            label(int)
        """
        assert isinstance(feats, np.ndarray), 'data should be ndarray.'
        assert (feats.shape[0]==1 or feats.ndim==1), 'data should be flatten data like(m,) or (1,m).'
        assert self.trained, 'model didnot trained, can not classify without pretrained params.'
        
        probs = self.sigmoid(self.W * feats)
        probs_to_label = probs > 0.5   # 概率转换成标签0,1
        return probs_to_label, probs

if __name__ == '__main__':
    
    
    import pandas as pd
    filename = 'testSet.txt'
    data = pd.read_csv(filename, sep='\t').values
    x = data[:,0:2]
    y = data[:,-1]
    
    logs = LogisticReg(x,y)
    logs.train()
    print('finish trained: %s'%str(logs.trained))
    print('W = ', logs.W)
    