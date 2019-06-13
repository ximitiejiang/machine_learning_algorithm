#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

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
        start = time.time()
        feats_with_one = np.concatenate([np.ones((len(self.feats),1)), self.feats], axis=1)  # (n_sample, 1+ n_feats) (1,x1,x2,..xn)
        labels = self.labels.reshape(-1, 1)   # (n_sample, 1)
        self.W = np.ones((feats_with_one.shape[1], 1)) # (m_feat, 1)
        self.losses = []        
        for i in range(n_epoch):
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn, 然后通过sigmoid函数转换为概率(0-1)
            # (n_sample, 1) 每个样本一个prob(0~1)，也就是作为2分类问题的预测概率
            w_x = np.dot(feats_with_one, self.W)  # w*x
            probs = self.sigmoid(w_x)     # (455,31) dot (31,1) -> (455,)
            
            loss = np.mean(-np.log(probs + 1e-06))  # loss = -log(P)
            self.losses.append([i,loss])
            gradient = - np.dot(feats_with_one.transpose(), (labels - probs))  # grad = -(y-y')*x, (3,n)dot((n,1)-(n,1))->(3,n)dot(n,1)->(3,)
            self.W -= alpha * gradient   # W(m,1), gradient(m,1)
            if i % 20 == 0:  # 每20个iter显示一次
                print('epoch: %d / %d, loss: %f, '%(i, n_epoch, loss))
        
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
    
    def evaluation(self, test_feats, test_labels):
        """评价整个验证数据集
        """
        correct = 0
        total_sample = len(test_feats)
        start = time.time()
        for feat, label in zip(test_feats, test_labels):
            pred_label, pred_prob = self.classify(feat)
            if int(pred_label) == int(label):
                correct += 1
        acc = correct / total_sample
        print('evaluation finished, with %f seconds.'%(time.time() - start))
        
        if test_feats.shape[1]==2: # 还没添加首列1，为2个特征
            self.vis_points_line(test_feats, test_labels)
        return acc
    
    def vis_loss(self, losses):
        """可视化损失"""
        assert losses is not None, 'can not visualize losses because losses is empty.'
        x = np.array(losses)[:,0]
        y = np.array(losses)[:,1]
        plt.subplot(1,2,1)
        plt.title('losses')
        plt.plot(x,y)
    
    def vis_points_line(self, feats, labels, W):
        """可视化二维点和分隔线(单组w)
        """
        assert feats.shape[1] == 2, 'feats should be 2 dimention data with 1st. column of 1.'
        assert len(W) == 3, 'W should be 3 values list.'
        
        feats_with_one = np.concatenate([np.ones((len(feats),1)), feats], axis=1)
        
        plt.subplot(1,2,2)
        plt.title('points and divide hyperplane')
        color = [c*64 + 64 for c in labels.reshape(-1)]
        plt.scatter(feats_with_one[:,1], feats_with_one[:,2], c=color)
        
        min_x = int(min(feats_with_one[:,1]))
        max_x = int(max(feats_with_one[:,1]))
        x = np.arange(min_x - 1, max_x + 1, 0.1)
        y = np.zeros((len(x),))
        for i in range(len(x)):
            y[i] = (-W[0,0] - x[i]*W[1,0]) / W[2,0]
        plt.plot(x, y, c='r')

    
if __name__ == '__main__':
    
    import pandas as pd
    filename = '2classes_data.txt'  # 一个简单的2个特征的二分类数据集
    data = pd.read_csv(filename, sep='\t').values
    x = data[:,0:2]
    y = data[:,-1]
    logs = LogisticReg(x,y)
    logs.train()
    print('finish trained: %s'%str(logs.trained))
    print('W = ', logs.W)
    sample = np.array([-1,1])
    label, prob = logs.classify(sample)
    print('label = %d, probility = %f'% (label, prob))
    
    