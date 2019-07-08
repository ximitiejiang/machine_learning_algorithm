#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""
import time
import numpy as np
from .base_model import BaseModel

class SoftmaxReg(BaseModel):
    def __init__(self, feats, labels):
        """ softmax reg algorithm lib, 可用于多分类的算法: 参考<机器学习算法-赵志勇, softmax regression>
        特点：支持二分类和多分类，模型参数可保存(n_feat+1, n_class)，支持线性可分数据
        
        softmax reg算法由一个线性模块(w0x0+w1x1+..wnxn)和一个非线性模块(softmax函数)组成一个函数
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)  注意，多分类样本标签必须是0-n，从0开头因为算法中需要用label从0开始定位取对应概率值
        """
        super().__init__(feats, labels)
    
    def softmax(self, x):
        """exp(wx)/sum(exp(wx))
        Args:
            x(array): (n_sample, k_class)
        Return:
            x_prob(array): (n_sample, k_class)
        """
        x_exp = np.exp(x)                    # (135,4), 这里要防止无穷大的产生
        x_sum = np.sum(x_exp, axis=1)        # (135,)
        x_sum_repeat = np.tile(x_sum.reshape(-1,1), (1, x.shape[1]))
        x_prob = x_exp / x_sum_repeat
        return x_prob
        
    def train(self, lr=0.001, n_epoch=500, batch_size=16):
        """feats(x1,x2,..xn) -> feats(1,x1,x2,..xn)
        Args:
            lr(float): 梯度下降步长
            n_epoch(inf): 循环训练轮数
        """
        assert batch_size <= len(self.labels), 'too big batch size, should be smaller than dataset size.'
        
        start = time.time()
        n_classes = len(set(self.labels))
        n_samples = len(self.feats)
        feats_with_one = np.concatenate([np.ones((n_samples,1)), self.feats], axis=1)  # (n_sample, 1+ n_feats) (1,x1,x2,..xn)
        labels = self.labels.reshape(-1, 1)   # (n_sample, 1)
        self.W = np.zeros((feats_with_one.shape[1], n_classes)) # (f_feat, c_classes)
        self.losses = []
        
        n_iter = n_epoch * (n_samples // batch_size)
        for i in range(n_iter):
            batch_feats, batch_labels = self.get_batch_data(
                feats_with_one, labels, batch_size=batch_size, type='shuffle')
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn, 然后通过softmax函数转换为概率(0-1)
            # (n_sample, 1) 每个样本一个prob(0~1)，也就是作为2分类问题的预测概率
            w_x = np.dot(batch_feats, self.W)       # w*x (b, c)
            probs = self.softmax(w_x)               # probability (b,c)
            
            # loss: 只提取正样本概率p转化为损失-log(p) 
            sum_loss = 0
            for sample in range(len(batch_labels)):
                sum_loss += -np.log(probs[sample, batch_labels[sample, 0]] / np.sum(probs[sample, :]))
            loss = sum_loss / len(batch_labels)  # average loss
            self.losses.append([i,loss])
            
            # vis text
            if i % 20 == 0 and i != 0:  # 每20个iter显示一次
                print('iter: %d / %d, loss: %f, '%(i, n_iter, loss))
            # gradient    
            _probs = - probs              # 取负号 -p
            for j in range(batch_size):
                label = batch_labels[j, 0]   # 提取每个样本的标签
                _probs[j, label] += 1   # 正样本则为1-p, 负样本不变依然是-p    
            gradient = - np.dot(batch_feats.transpose(), _probs)  # (135,3).T * (135,4) -> (3,4), grad = -x*(I-y')   
            # update Weights
            self.W -= lr * gradient * (1/batch_size)   # W(3,4) - (a/n)*(3,4), 因前面计算梯度时我采用的负号，这里就应该是w = w-lr*grad
        
        self.vis_loss(self.losses)
        if self.feats.shape[1] == 2:
            for j in range(self.W.shape[1]):   # w (3,4)代表3个特征，4个类别
                w = self.W[:,j].reshape(-1,1)
                self.vis_points_line(self.feats, self.labels, w)
        
        print('training finished, with %f seconds.'%(time.time() - start))
        
        self.trained = True
        # prepare model_dict for saving
        self.model_dict['model_name'] = 'SoftmaxReg'
        self.model_dict['W'] = self.W
        return self
        
    def predict_single(self, single_sample_feats):
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
        probs = self.softmax(np.dot(single_sample_feats, self.W))  # w*x
        label = np.argmax(probs)   # 获得最大概率所在位置，即标签(所以也要求标签从0开始 0~n)
#        label_prob = probs[0,label]
        return label
    


