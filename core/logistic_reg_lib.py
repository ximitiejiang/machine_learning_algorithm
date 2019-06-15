#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""

import numpy as np
import time
from .base import BaseModel

class LogisticReg(BaseModel):
    def __init__(self, feats, labels):
        """ logistic reg algorithm lib, 当前只支持2分类
        特点：支持二分类，模型参数可保存(n_feat+1, 1)，支持线性可分数据
        
        logistic reg算法由一个线性模块(w0x0+w1x1+..wnxn)和一个非线性模块(sigmoid函数)组成一个函数
        用输入特征feats和labels来训练这个模块，得到一组(w0,w1,..wn)的模型，可用来进行二分类问题的预测，但不能直接用于多分类问题
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)
        """
        super().__init__(feats, labels)
    
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))    
    
    
    def train(self, alpha=0.001, n_epoch=500, batch_size=64):
        """feats(x1,x2,..xn) -> feats(1,x1,x2,..xn)
        Args:
            alpha(float): 梯度下降步长
            n_epoch(inf): 循环训练轮数
        """
        assert batch_size <= len(self.labels), 'too big batch size, should be smaller than dataset size.'
        start = time.time()
        feats_with_one = np.concatenate([np.ones((len(self.feats),1)), self.feats], axis=1)  # (n_sample, 1+ n_feats) (1,x1,x2,..xn)
        labels = self.labels.reshape(-1, 1)   # (n_sample, 1)
        self.W = np.ones((feats_with_one.shape[1], 1)) # (m_feat, 1)
        self.losses = []
        
        n_samples = len(self.feats)
        n_iter = n_epoch * (n_samples // batch_size)        
        for i in range(n_iter):
            batch_feats, batch_labels = self.get_batch_data(
                feats_with_one, labels, batch_size=batch_size, type='shuffle')
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn, 然后通过sigmoid函数转换为概率(0-1)
            # (n_sample, 1) 每个样本一个prob(0~1)，也就是作为2分类问题的预测概率
            w_x = np.dot(batch_feats, self.W)  # w*x
            probs = self.sigmoid(w_x)     # (455,31) dot (31,1) -> (455,)
            
            loss = np.mean(-np.log(probs + 1e-06))  # loss = -log(P)
            self.losses.append([i,loss])
            if i % 20 == 0 and i != 0:  # 每20个iter显示一次
                print('iter: %d / %d, loss: %f'%(i, n_iter, loss))
                
            gradient = - np.dot(batch_feats.transpose(), (batch_labels - probs))  # grad = -(y-y')*x, (3,n)dot((n,1)-(n,1))->(3,n)dot(n,1)->(3,)
            self.W -= alpha * gradient   # W(m,1), gradient(m,1)
        
        self.vis_loss(self.losses)
        if self.feats.shape[1] == 2:  # 如果是二维特征则显示分割直线
            self.vis_points_line(self.feats, labels, self.W)
        
        self.trained = True
        print('training finished, with %f seconds.'%(time.time() - start))
        
    def classify(self, single_sample_feats):
        """ 单样本预测
        Args:
            data(numpy): (1, m_feats) or (m_feats,)，如果是(m,n)则需要展平
            k(int): number of neibours
        Returns:
            label(int)
        """
        assert isinstance(single_sample_feats, np.ndarray), 'data should be ndarray.'
        assert (single_sample_feats.shape[0]==1 or single_sample_feats.ndim==1), 'data should be flatten data like(m,) or (1,m).'
        assert self.trained, 'model didnot trained, can not classify without pretrained params.'
        single_sample_feats = np.concatenate([np.array([1]), single_sample_feats]).reshape(1,-1)
        probs = self.sigmoid(np.dot(single_sample_feats, self.W))  # w*x
        probs_to_label = ((probs > 0.5)+0)[0,0]   # 概率转换成标签0,1: 大于0.5为True, +0把True转换成1, 提取[0,0]
        return probs_to_label, probs[0,0]

    
    