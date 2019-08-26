#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:01:00 2019

@author: ubuntu
"""
import numpy as np

# %% 损失函数
class Loss():
    """创建一个损失基类，仅用来汇总共有参数"""
    def loss(self, y, p):
        raise NotImplementedError()
    
    def gradient(self, y, p):
        raise NotImplementedError()
    
    def acc(self, y, p):  # 该acc函数可用于每个epoch输出一个acc，用于评价模型是否过拟合或者欠拟合：train_acc vs test_acc，从而在训练过程中就能评估出来。
        p = np.argmax(p, axis=1)  # (1280,)
        y = np.argmax(y, axis=1)  # (1280,)
        acc = np.sum(p == y, axis=0) / len(y)
        return acc


class CrossEntropy(Loss):
    """交叉熵损失函数，通过评价两个分布y,y'的相关性来评价分类结果的准确性，而相关性指标采用交叉熵"""
    def __init__(self): 
        pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)  # 计算损失loss = -(y*log(y') + (1-y)*log(1-y'))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)  # 计算该级梯度grad = loss' = -y/y' + (1-y)/(1-y')


class SquareLoss(Loss):
    """平方损失函数，通过评价两个分布y,y'的空间距离来评价回归结果的准确性。
    平方损失类似于所谓的mse()损失函数，只不过mse()是对平方损失进行缩减得到一个但损失值，
    而平方损失跟交叉熵损失都是直接输出包含每个元素的损失矩阵。
    """
    def __init__(self): 
        pass
    
    def loss(self, y, p):
        loss_result = 0.5 * np.power((y - p), 2)  # 增加一个0.5是为了让梯度结果更简洁，直接抵消平方求导产生的2
        return loss_result
    
    def gradient(self, y, p):
        grad_result = -(y - p)
        return grad_result
    
    
#class SquareLoss(Loss):
#    """平方损失函数，通过评价两个分布y,y'的空间距离来评价回归结果的准确性。
#    平方损失类似于所谓的mse()损失函数，只不过mse()是对平方损失进行缩减得到一个但损失值，
#    而平方损失跟交叉熵损失都是直接输出包含每个元素的损失矩阵。
#    """
#    def __init__(self, regularization=None): # 正则参数可以输入None, l1, l2
#        if regularization is None:
#            self.regularization = None
#        elif regularization == 'l1':
#            self.regularization = l1_regularization()
#        elif regularization == 'l2':
#            self.regularization = l2_regularization()
#    
#    def loss(self, y, p, w=None):
#        if self.regularization is None:
#            loss_result = 0.5 * np.power((y - p), 2)  # 增加一个0.5是为了让梯度结果更简洁，直接抵消平方求导产生的2
#        else:
#            loss_result = 0.5 * np.power((y - p), 2) + self.regularization(w)
#        
#        return loss_result
#    
#    def gradient(self, y, p, w=None):
#        if self.regularization is None:
#            grad_result = -(y - p)
#        else:
#            grad_result =  -(y - p) + self.regularization.grad(w)
#        return grad_result



    
    