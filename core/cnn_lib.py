#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:02:41 2019

@author: ubuntu
"""
import numpy as np

class CNN():
    """纯python版本的神经网络模型"""
    def __init__(self, optimizer, loss): # optimizer传入一个对象，
        self.optimizer = optimizer
        self.loss_function = loss()
        self.layers = []
    
    def add(self, layer):
        if self.layers:
            layer.set_input_shape(shape = self.layers[-1].output_shape())
        if hasattr(layer, "initialize"):
            layer.initialize(optimizer=self.optimizer)
            
        self.layers.append(layer)
        
    def train_on_batch(self):
        pass
    
    def test_on_batch(self):
        pass
    
    def train(self, feats, labels, n_epoch, batch_size):
        pass
    
    def _forward_pass(self, feats, training=True):
        pass
    
    def _backward_pass(self, loss_grad):
        
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)
    
    def predict(self, feat):
        return self._forward_pass(feat, training=True)

    

# %% 基础层
class Layer():
    def set_input_shape(self, shape):
        self.input_shape = shape
    
    def layer_name(self):
        return self.__class__.__name__
    
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None


# %% 激活函数
class Sigmoid():
    def __int__():
        pass
    
    def __call__():
        pass
    
    def grad():
        pass

class Softmax():
    def __int__():
        pass
    
    def __call__():
        pass
    
    def grad():
        pass
    

# %% 损失函数
class CrossEntropy:
    def __init__(self):
        pass
    
    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    def grad(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y / p - (1 - y) / (1 - p))
    
    
# %% 优化器
class Adam:
    """Adam优化器：对梯度的一阶矩估计(梯度的均值)和二阶距估计(梯度的未中心化方差)进行综合考虑来更新步长。
    Args:
        lr：学习率
        b1/b2: 矩估计的指数衰减率 
    """
    def __init__(self, lr=0.001, b1=0.9, b2=0.999):
        self.lr = lr
        self.eps = 1e-8
        self.m = None
        self.v = None
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
        return w - self.w_update
    
    
if __name__ == "__main__":
    
    optimizer = Adam()
    
    cnn = CNN(optimizer=optimizer, loss=CrossEntropy)
        
    