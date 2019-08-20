#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""

import numpy as np
import time
import math
import torch
import torch.nn as nn
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


class LogisticReg(BaseModel):
    def __init__(self, feats, labels, lr=0.001, n_epoch=500, batch_size=-1):
        """ logistic reg algorithm lib, 当前只支持2分类
        特点：支持二分类，模型参数可保存(n_feat+1, 1)，支持线性可分数据
        
        logistic reg算法由一个线性模块(w0x0+w1x1+..wnxn)和一个非线性模块(sigmoid函数)组成一个函数
        用输入特征feats和labels来训练这个模块，得到一组(w0,w1,..wn)的模型，可用来进行二分类问题的预测，但不能直接用于多分类问题
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)
        """
        super().__init__(feats, labels)       
        self.lr = lr
        self.n_epoch = n_epoch
        self.batch_size = batch_size
    
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))    
    
    def weight_init(self, n_feats):
        limit = 1/ np.sqrt(n_feats)
        w = np.random.uniform(-limit, limit, (n_feats, ))  # 输出(n,)
        return w
    
    def train(self):
        """feats(x1,x2,..xn) -> feats(1,x1,x2,..xn)
        """
        assert self.batch_size <= len(self.labels), 'too big batch size, should be smaller than dataset size.'
        start = time.time()
        feats_with_one = np.concatenate([np.ones((len(self.feats),1)), self.feats], axis=1)  # (n_sample, 1+ n_feats) (1,x1,x2,..xn)
        labels = self.labels.reshape(-1)   # (n_sample,)
#        self.W = np.ones((feats_with_one.shape[1], 1)) # (m_feat, 1)
        self.W = self.weight_init(feats_with_one.shape[1])  # return (n_feat, )
        self.losses = []
        
        n_samples = len(self.feats)
        n_iter = self.n_epoch if self.batch_size==-1 else self.n_epoch * (n_samples // self.batch_size)        
        for i in range(n_iter):
            batch_feats, batch_labels = self.get_batch_data(
                feats_with_one, labels, batch_size=self.batch_size, type='shuffle')
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn, 然后通过sigmoid函数转换为概率(0-1)
            # (n_sample, 1) 每个样本一个prob(0~1)，也就是作为2分类问题的预测概率
            w_x = np.dot(batch_feats, self.W)  # w*x
            y_probs = self.sigmoid(w_x)     # (455,31) dot (31,1) -> (455,)
            
            # 求损失loss=-(ylogy' + (1-y)log(1-y'))：(79,)*(79,) + （79,)*(79,) -> (79,)
            # 注意：预测值y_probs如果接近0，则log(y_probs)趋近负无穷
            iter_losses = cross_entropy(y_probs, batch_labels)
            loss = np.mean(iter_losses)
            self.losses.append([i,loss])
            if i % 20 == 0 and i != 0:  # 每20个iter显示一次
                print('iter: %d / %d, loss: %f'%(i, n_iter, loss))
            
            # 求梯度：grad = -(y-y')*x
            gradient = - np.dot((batch_labels - y_probs), batch_feats)  # (79,)dot(79,3)->(1,79)dot(79,3)->(1,3)->(3,)
            self.W -= self.lr * gradient   # W(m,1), gradient(m,1)
        
        self.vis_loss(self.losses)
        if self.feats.shape[1] == 2:  # 如果是二维特征则显示分割直线
            self.vis_points_line(self.feats, labels, self.W)
        
        print('training finished, with %f seconds.'%(time.time() - start))
        
        self.trained = True
        # prepare model_dict for saving
        self.model_dict['model_name'] = 'LogisticReg'
        self.model_dict['W'] = self.W

        
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
        probs = self.sigmoid(np.dot(single_sample_feats, self.W))  # w*x
        probs_to_label = ((probs[0] > 0.5)+0)   # 概率转换成标签0,1: 大于0.5为True, +0把True转换成1, 提取[0,0]
        return probs_to_label

    
class LogisticReg_autograd:
    """只更换了pytorch底层的自动求导模块来替代手动求导
    input: 跟原有机器学习模型保持一致，在内部进行tensor的转化
        feats
        labels
        lr
        n_epoch
        batch_size
        device:  torch.device("cpu") or torch.device("cuda:0")
    """
    def __init__(self, feats, labels, lr=0.001, n_epoch=500, batch_size=-1, 
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.feats = torch.tensor(feats, dtype=torch.float32, device=device)
        self.labels = torch.tensor(labels, dtype=torch.float32, device=device)
        self.lr = lr
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.device = device
        
        self.num_features = feats.shape[1]
        
        self.weights = torch.zeros(self.num_features, 1, dtype= torch.float32, device=device,requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float32, device = device, requires_grad=True)
    
    def sigmoid(self, x):
        return 1. / (1. + torch.exp(-x))
    
    
    def forward(self, x):
        linear = torch.add(torch.mm(x, self.weights), self.bias)
        probs = self.sigmoid(linear)
        return probs
        
    def losses(self, y, y_prob):
        tmp1 = torch.mm(y.view(-1,1), torch.log(y_prob))
        tmp2 = torch.mm((1 - y).view(1, -1), torch.log(1-y_prob))
        return - (tmp1 + tmp2)  # loss = -(y*logy' + (1-y)*log(1-y'))
    
    def evaluation(self, x, y):
        prob = self.forward(x)
        labels = (prob > 0.5) + 0.
        acc = torch.sum(labels.view(-1)).float() / y.size()[0]
        return acc
    
    def vis_boundary(self):
        pass
    
    def train(self):
        # 暂时不支持batch size的设置            
        for epoch in range(self.n_epoch):
            # 前向计算
            y_prob = self.forward(self.feats)
            # 计算损失
            loss = self.losses(self.labels, y_prob)
            # 计算梯度
            loss.backward()   # 会自动计算每个w的梯度(在计算图中沿着loss的计算路径反向计算每个自变量的梯度)
            # 更新权重
            wt = self.weights.detach()          #
            wt -= self.lr * self.weights.grad   #
            
            bs = self.bias.detach()
            bs -= self.lr * self.bias.grad
            
            # 梯度清零
            self.weights.grad.zero_()
            self.bias.grad.zero_()
            
            # 显示
            print("epoch: %d / %d:" % (epoch, self.n_epoch))
            print(" | training acc: %.3f" % self.evaluation(self.feats, self.labels))
            print(" | loss: %.3f" % loss)
            

class LogisticReg_pytorch(nn.Module):
    """完全采用pytorch自带各种层、损失函数来搭建逻辑回归
    """
    def __init__(self, feats, labels,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.feats = torch.tensor(feats, dtype=torch.float32, device=device)
        self.labels = torch.tensor(labels, dtype=torch.float32, device=device)
        
        self.num_features = feats.shape[1]
        self.linear = torch.nn.Linear()
        self.linear.weight.detach().zero_()  # 直接用pytorch的层初始化参数
        self.linear.bias.detach().zero_()
        
    def forward(self, x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs
    

import tensorflow as tf    
class LogisticReg_tensorflow():
    def __init__():
        pass