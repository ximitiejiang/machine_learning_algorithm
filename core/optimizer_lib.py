#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:16:54 2019

@author: ubuntu
"""
import numpy as np

# %% 优化器(用于更新权重)
class Adam():
    """Adam优化器：对梯度的一阶矩估计(梯度的均值)和二阶距估计(梯度的未中心化方差)进行综合考虑来更新步长。
    Args:
        lr：学习率
        b1/b2: 矩估计的指数衰减率 
    """
    def __init__(self, lr=0.001, b1=0.9, b2=0.999):
        self.lr = lr
        self.eps = 1e-8
        self.m = None  # 梯度的一阶矩
        self.v = None  # 梯度的二阶距
        self.b1 = b1
        self.b2 = b2
    
    def update(self, w, grad):
        if self.m is None:
            self.m = np.zeros(w.shape)  # 初始化一阶矩为0
            self.v = np.zeros(w.shape)  # 初始化二阶距为0
        self.m = self.b1 * self.m + (1 - self.b1) * grad               # 一阶矩迭代更新mt = b1*mt-1 + (1-b1)*g
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad, 2)  # 二阶距迭代更新vt = b2*bt-1 + (1-b2)*g^2
        
        m_hat = self.m / (1 - self.b1)    # 计算偏置修正一阶矩mt' = mt/(1-b1)
        v_hat = self.v / (1 - self.b2)    # 计算偏置修正二阶距vt' = vt/(1-b2)
        self.w_update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # 更新参数w = w - lr*mt'/(sqrt(vt') + eps)
        w -= self.w_update
        return w


class SGD():
    """普通SGD梯度下降算法"""
    def __init__(self, lr=0.001):
        self.lr = lr
    
    def update(self, w, grad):
        w -= self.lr * grad
        return w


class SGDM():
    """SGDM梯度下降算法-带动量M的SGD算法"""
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.tmp_w = None
    
    def update(self, w, grad):
        if self.tmp_w is None:
            self.tmp_w = np.zeros_like(w)
        self.tmp_w = self.momentum * self.tmp_w + (1 - self.momentum) * grad  # 计算更新m
        w -= self.lr * self.tmp_w
        return w
    