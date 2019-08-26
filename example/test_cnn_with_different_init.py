#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:20:23 2019

@author: ubuntu
"""

from core.neural_network_lib import NeuralNetwork, Linear, Activation 
import numpy as np

"""
权重初始化的最佳实践：
1. 必须随机初始化权重，防止权重均一化：
2. 必须尽可能小的初始化权重，防止权重值过大造成过拟合：
3. 必须
"""


class MLP5L(NeuralNetwork):
    """基于神经网络结构的多层卷积神经网络"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
        
        self.add(Linear(in_features=100, out_features=100))
        self.add(Activation('relu'))
        self.add(Linear(in_features=100, out_features=100))
        self.add(Activation('relu'))
        self.add(Linear(in_features=100, out_features=100))
        self.add(Activation('relu'))
        self.add(Linear(in_features=100, out_features=100))
        self.add(Activation('relu'))
        self.add(Linear(in_features=100, out_features=10))
        self.add(Activation('softmax'))
    
    def forward_pass(self, x):
        """重写前向计算，提取激活值"""
        self.activations = []
        for i, layer in enumerate(self.layers):
            layer_output = layer.forward_pass(x)
            x = layer_output
            
            if i % 2 == 1: # 奇数层
                self.activations.append(layer_output) # 保存每层激活值
        return layer_output
    

        
if __name__ == "__main__":
    x = np.random.randn(1000, 100)  # 1000个样本，特征数量100个
    
    model = MLP5L()