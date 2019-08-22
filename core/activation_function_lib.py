#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:57:22 2019

@author: ubuntu
"""

import numpy as np

# %%激活函数
       
class Sigmoid():
    """sigmoid函数的计算及梯度计算: 对sigmoid函数直接可求导"""
    def __call__(self, x):
        return 1 / (1 + np.exp(-x)) 
    
    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
    
    
class Softmax():
    """softmax函数的计算及梯度计算：对softmax函数直接可求导"""
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 在np.exp(x - C), 相当于对x归一化，防止因x过大导致exp(x)无穷大使softmax输出无穷大 
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def gradient(self, x):
        y_p = self.__call__(x)  # 先求得输出y'
        return y_p * (1 - y_p)  # grad = y'*(1-y')
    
    
class Relu():
    """relu函数的计算及梯度计算：对relu函数直接可求导"""
    def __call__(self, x):
        return np.where(x>=0, x, 0)   # 注意这里是矩阵操作，不能用普通的判断，需要用np.where做矩阵级判断
    
    def gradient(self, x):
        return np.where(x>=0, 1, 0)


class LeakyRelu():
    """leaky relu函数的计算及梯度计算：解决relu在x<0时梯度消失为0的问题"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, x):
        return np.where(x>=0, x, self.alpha * x)
    
    def gradient(self, x):
        return np.where(x>=0, 1, self.alpha)

class Elu():
    """Elu函数的计算及梯度计算"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def __call__(self, x):
        return np.where(x>=0, x, self.alpha * (np.exp(x) - 1))
    
    def gradient(self, x):
        return np.where(x>=0, 1, self.__call__(x) + self.alpha)
