#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:01:00 2019

@author: ubuntu
"""
import numpy as np

# %% 损失函数
class CrossEntropy():
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