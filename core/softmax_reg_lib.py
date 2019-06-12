#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""

import numpy as np
import os
import time
from tqdm import tqdm

class SoftmaxReg():
    def __init__(self, feats, labels):
        """ softmax reg algorithm lib, 可用于多分类的算法: 参考<机器学习算法-赵志勇, softmax regression>
        softmax reg算法由一个线性模块(w0x0+w1x1+..wnxn)和一个非线性模块(softmax函数)组成一个函数
        用输入特征feats和labels来训练这个模块，得到一组(w0,w1,..wn)的模型，可用来进行二分类问题的预测，但不能直接用于多分类问题
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)  注意，多分类样本标签必须是0-n，从0开头因为算法中需要用label从0开始定位取对应概率值
        """
        assert feats.ndim ==2, 'the feats should be (n_samples, m_feats), each sample should be 1-dim flatten data.'
        self.feats = feats
        self.labels = labels.astype(np.int8)
        self.trained = False
    
    def softmax(self, x):
        """exp(wx)/sum(exp(wx))
        Args:
            x(array): (n_sample, k_class)
        Return:
            x_prob(array): (n_sample, k_class)
        """
        x_exp = np.exp(x)                    # (135,4)
        x_sum = np.sum(x_exp, axis=1)        # (135,)
        x_sum_repeat = np.tile(x_sum.reshape(-1,1), (1,4))
        x_prob = x_exp / x_sum_repeat
        return x_prob
            
    def train(self, alpha=0.001, n_epoch=500):
        """feats(x1,x2,..xn) -> feats(1,x1,x2,..xn)
        Args:
            alpha(float): 梯度下降步长
            n_epoch(inf): 循环训练轮数
        """
        start = time.time()
        n_classes = len(set(self.labels))
        n_samples = len(self.feats)
        feats = np.concatenate([np.ones((n_samples,1)), self.feats], axis=1)  # (n_sample, 1+ n_feats) (1,x1,x2,..xn)
        labels = self.labels.reshape(-1, 1)   # (n_sample, 1)
        self.W = np.ones((feats.shape[1], n_classes)) # (m_feat, k_classes)
        
        for i in tqdm(range(n_epoch)):
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn, 然后通过softmax函数转换为概率(0-1)
            # (n_sample, 1) 每个样本一个prob(0~1)，也就是作为2分类问题的预测概率
            w_x = np.dot(feats, self.W)   # w*x  (135, 4) (n_sample, k_class)
            probs = self.softmax(w_x)     # probability (135,4)
            _probs = - probs              # 取负号 
            for j in range(n_samples):
                label = labels[j, 0]   # 提取每个样本的标签
                _probs[j, label] = 1 + _probs[j, label]   # 正样本则为1-p, 负样本则为-p
                
            gradient = - np.dot(feats.transpose(), _probs)  # (135,3).T * (135,4) grad = -x*(I-y')，其中I=1(正样本)，I=0(负样本)   
            self.W -= alpha/n_samples * gradient   # W(m,1), gradient(m,1)
        self.trained = True
        print('training finished, with %f seconds.'%(time.time() - start))
        
    def classify(self, feats):
        """ 单样本预测
        Args:
            data(numpy): (1, m_feats) or (m_feats,)，如果是(m,n)则需要展平
            k(int): number of neibours
        Returns:
            label(int)
        """
        assert isinstance(feats, np.ndarray), 'data should be ndarray.'
        assert (feats.shape[0]==1 or feats.ndim==1), 'data should be flatten data like(m,) or (1,m).'
        assert self.trained, 'model didnot trained, can not classify without pretrained params.'
        feats = np.concatenate([np.array([1]), feats]).reshape(1,-1)
        probs = self.softmax(np.dot(feats, self.W))  # w*x
        label = np.argmax(probs)   # 获得最大概率所在位置，即标签(所以也要求标签从0开始 0~n)
        label_prob = probs[0,label]
        return label, label_prob
    
    def evaluation(self, test_feats, test_labels):
        """评价一组数据集"""
        correct = 0
        total_sample = len(test_feats)
        start = time.time()
        for feat, label in tqdm(zip(test_feats, test_labels)):
            pred_label, pred_prob = self.classify(feat)
            if int(pred_label) == int(label):
                correct += 1
        acc = correct / total_sample
        print('evaluation finished, with %f seconds.'%(time.time() - start))
        return acc

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    filename = '4classes_data.txt'  # 一个简单的2个特征的二分类数据集
    data = pd.read_csv(filename, sep='\t').values
    x = data[:,0:2]
    y = data[:,-1]
    plt.scatter(x[:,0],x[:,1])
    soft = SoftmaxReg(x,y)
    soft.train()
    print('finish trained: %s'%str(soft.trained))
    print('W = ', soft.W)
    sample = np.array([-1,1])
    label, prob = soft.classify(sample)
    print('label = %d, probility = %f'% (label, prob))