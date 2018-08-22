#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 08:51:50 2018

@author: suliang
"""
import numpy as np

def AND(x1, x2):  # 与门
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    temp = sum(x * w) + b # 对应位置乘（只有mat才跟matlab一样是点积）
    if temp <=0:
        return 0
    else:
        return 1
    
def NAND(x1, x2):  # 与非门
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    temp = sum(x * w) + b # 对应位置乘（只有mat才跟matlab一样是点积）
    if temp <=0:
        return 0
    else:
        return 1
    
def OR(x1,x2):  # 或门
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    temp = sum(x * w) + b # 对应位置乘（只有mat才跟matlab一样是点积）
    if temp <=0:
        return 0
    else:
        return 1
    
def XOR(x1, x2):  # 异或门：不能用单层线性感知机实现
    x = np.array([x1, x2])
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    temp = AND(s1,s2)
    return temp


# ------def for NN--------------
def sigmoid(x):     # 逻辑函数
    return 1.0/(1+np.exp(-x))


def identity_function(x):   # 恒等函数
    return x


def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1      # 中间层激活函数都是逻辑函数
    z1 = sigmoid(a1)    
    
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)     # 输出层的激活函数是恒等函数

    return y

    
network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)
    
    

    