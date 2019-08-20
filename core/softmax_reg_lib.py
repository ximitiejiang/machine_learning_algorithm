#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""
import time
import numpy as np
from .base_model import BaseModel

def cross_entropy(y_preds, y_labels):
    """二值交叉熵: loss = -(y*log(y') + (1-y)log(1-y'))，其中y为概率标签(0或1)，y'为预测概率(0~1)
    其中当输入是一维的数据，则为二分类交叉熵；输入是二维的数据，则为多分类交叉熵(等效于每个位做二分类计算)
    Args:
        y_preds: (m,)or(m,k)
        y_labels: (m,)or(m,k)
    Return:
        loss: (m,)or(m,k)
    """
    loss = - (y_labels * np.log(y_preds) + (1 - y_labels)*np.log(1 - y_preds)) # (m,)
    return loss

class SoftmaxReg(BaseModel):
    def __init__(self, feats, labels, lr=0.001, n_epoch=500, batch_size=-1):
        """ softmax reg algorithm lib, 可用于多分类的算法: 参考<机器学习算法-赵志勇, softmax regression>
        特点：支持二分类和多分类，模型参数可保存(n_feat+1, n_class)，支持线性可分数据
        
        softmax reg算法由一个线性模块(w0x0+w1x1+..wnxn)和一个非线性模块(softmax函数)组成一个函数
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)  注意，多分类样本标签必须是0-n，从0开头因为算法中需要用label从0开始定位取对应概率值
        """
        super().__init__(feats, labels)
        self.lr = lr
        self.n_epoch = n_epoch
        self.batch_size = batch_size
    
    def softmax(self, x):
        """exp(wx)/sum(exp(wx))
        Args:
            x(array): (n_sample, k_class)
        Return:
            x_prob(array): (n_sample, k_class)
        """
        x_exp = np.exp(x)                    # (135,4), 这里要防止无穷大的产生, 所以x最好standardlize
        x_sum = np.sum(x_exp, axis=1)        # (135,)
        x_sum_repeat = np.tile(x_sum.reshape(-1,1), (1, x.shape[1]))
        x_prob = x_exp / x_sum_repeat
        return x_prob
        
    def train(self):
        """feats(x1,x2,..xn) -> feats(1,x1,x2,..xn)
        Args:
            lr(float): 梯度下降步长
            n_epoch(inf): 循环训练轮数
        """
        assert self.batch_size <= len(self.labels), 'too big batch size, should be smaller than dataset size.'
        
        start = time.time()
        # 计算类别数：普通标签和独热标签区别对待
        if self.labels.ndim > 1:
            n_classes = self.labels.shape[1]
            labels = self.labels
        else:
            n_classes = len(set(self.labels))
            labels = self.labels.reshape(-1, 1)   # (n_sample, 1)
        
        n_samples = len(self.feats)
        feats_with_one = np.concatenate([np.ones((n_samples,1)), self.feats], axis=1)  # (n_sample, 1+ n_feats) (1,x1,x2,..xn)
        
        self.W = np.ones((feats_with_one.shape[1], n_classes)) # (f_feat, c_classes)
        self.losses = []
        
        n_iter = self.n_epoch if self.batch_size==-1 else self.n_epoch * (n_samples // self.batch_size)
        for i in range(n_iter):
            batch_feats, batch_labels = self.get_batch_data(
                feats_with_one, labels, batch_size=self.batch_size, type='shuffle')
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn, 然后通过softmax函数转换为概率(0-1)
            # (n_sample, 1) 每个样本一个prob(0~1)，也就是作为2分类问题的预测概率
            w_x = np.dot(batch_feats, self.W)       # w*x (b, c)
            y_probs = self.softmax(w_x)             # (n_sample, n_feats)列变为n_feats相当于对独热编码标签的每一个类别进行二分类预测。
            
            # 求损失loss: 传入预测概率和标签概率，采用交叉熵按位
            iter_losses = cross_entropy(y_probs, batch_labels)  # (m,4) (m,4) ->(m,4)
            loss = np.mean(iter_losses)  
            self.losses.append([i,loss])
            if i % 20 == 0 and i != 0:  # 每20个iter显示一次
                print('iter: %d / %d, loss: %f, '%(i, n_iter, loss))
            
            # 求梯度：gradient: grad = -X*(I{y=j} - y')，其中I为独热标签y，y'为预测概率
            #gradient = - np.dot(batch_feats.transpose(), (1-y_probs))  # (70,3).T * (70,4) -> (3,4), grad = -x*(I-y')   
            gradient = - np.dot(batch_feats.T, (batch_labels - y_probs))
            # update Weights
            self.W -= self.lr * (1/n_samples) * gradient  # (3,4)
        
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
    


