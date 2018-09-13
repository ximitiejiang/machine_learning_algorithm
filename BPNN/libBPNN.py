#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:37:25 2018

@author: suliang
"""

# BPNN = back propagation neural network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_name):
    # 1、获取特征
    feature_data = []
    label_tmp = []
    with open(file_name) as f:
        for line in f.readlines():
            feature_tmp = []
            lines = line.strip().split("\t")
            label_tmp.append(int(lines[-1])) # 提取每一行的最后一个值为标签 
        
            for i in range(len(lines) - 1):
                feature_tmp.append(float(lines[i]))  # 提取每一行前i-1个值为数据
            feature_data.append(feature_tmp)
    
    # 对分类标签进行one-hot编码
    m = len(label_tmp)
    n_class = len(set(label_tmp))  # 得到类别的个数
    label_data = np.mat(np.zeros((m, n_class)))
    for i in range(m):
        label_data[i, label_tmp[i]] = 1
    
    return np.mat(feature_data), label_data, n_class


def sig(x):
    '''Sigmoid函数
    input:  x(mat/float):自变量，可以是矩阵或者是任意实数
    output: Sigmoid值(mat/float):Sigmoid函数的值
    '''
    return 1.0 / (1 + np.exp(-x))

def partial_sig(x):
    '''Sigmoid导函数的值
    input:  x(mat/float):自变量，可以是矩阵或者是任意实数
    output: out(mat/float):Sigmoid导函数的值
    '''
    m, n = np.shape(x)
    out = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            out[i, j] = sig(x[i, j]) * (1 - sig(x[i, j]))  # 对sigmoid求导输出=sig*(1-sig)
    return out


# reLU激活函数
def reLU():
    pass

def partial_reLU():
    pass

# 多分类激活函数
def softmax():
    pass

def hidden_in(feature, w0, b0):
    '''计算隐含层的输入
    input:  feature(mat):特征
            w0(mat):输入层到隐含层之间的权重
            b0(mat):输入层到隐含层之间的偏置
    output: hidden_in(mat):隐含层的输入
    '''
    m = np.shape(feature)[0]
    hidden_in = feature * w0  # X * W0
    for i in range(m):
        hidden_in[i, ] += b0
    return hidden_in

def hidden_out(hidden_in):
    '''隐含层的输出
    input:  hidden_in(mat):隐含层的输入
    output: hidden_output(mat):隐含层的输出
    '''
    hidden_output = sig(hidden_in)
    return hidden_output;


def predict_in(hidden_out, w1, b1):
    '''计算输出层的输入
    input:  hidden_out(mat):隐含层的输出
            w1(mat):隐含层到输出层之间的权重
            b1(mat):隐含层到输出层之间的偏置
    output: predict_in(mat):输出层的输入
    '''
    m = np.shape(hidden_out)[0]
    predict_in = hidden_out * w1  # a1 *  W1
    for i in range(m):
        predict_in[i, ] += b1
    return predict_in
    
def predict_out(predict_in):
    '''输出层的输出
    input:  predict_in(mat):输出层的输入
    output: result(mat):输出层的输出
    '''
    result = sig(predict_in)  # 2分类问题，用sigmoid()输出0-1
    return result


def get_predict(feature, w0, w1, b0, b1):
    '''计算最终的预测
    input:  feature(mat):特征
            w0(mat):输入层到隐含层之间的权重
            b0(mat):输入层到隐含层之间的偏置
            w1(mat):隐含层到输出层之间的权重
            b1(mat):隐含层到输出层之间的偏置
    output: 预测值
    '''
    return predict_out(predict_in(hidden_out(hidden_in(feature, w0, b0)), w1, b1))    


def get_cost(cost):
    '''计算当前损失函数的值
    input:  cost(mat):预测值与标签之间的差
    output: cost_sum / m (double):损失函数的值
    '''
    m,n = np.shape(cost)
    
    cost_sum = 0.0
    for i in range(m):
        for j in range(n):
            cost_sum += cost[i,j] * cost[i,j]
    return cost_sum / m


# 创建一个单隐藏层的多层感知机，该隐藏层节点数可自定义，输出层的节点数可自定义
def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    from math import sqrt
    ''' 计算隐含层的输入
        input:  feature(mat):特征
            label(mat):标签
            n_hidden(int):隐含层的节点个数
            maxCycle(int):最大的迭代次数
            alpha(float):学习率
            n_output(int):输出层的节点个数
        output: w0(mat):输入层到隐含层之间的权重
            b0(mat):输入层到隐含层之间的偏置
            w1(mat):隐含层到输出层之间的权重
            b1(mat):隐含层到输出层之间的偏置
    '''
    m, n = np.shape(feature)
    # 初始化: 假定的逻辑是：输入层不算一层，第一层从隐藏层算起
    # 所以w0,b0都是输入层的参数，而隐藏层之后的才取名w1,b1
    # 如何初始化是很困难的事情: 建议的？
    
    # w0首生成[0,1]之间均匀分布数，再乘以sqrt(6/(n+n_hidden))
    w0 = np.mat(np.random.rand(n, n_hidden))
    w0 = w0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - \
         np.mat(np.ones((n, n_hidden))) * \
         (4.0 * sqrt(6) / sqrt(n + n_hidden))
    
    b0 = np.mat(np.random.rand(1, n_hidden))
    b0 = b0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - \
         np.mat(np.ones((1, n_hidden))) * \
         (4.0 * sqrt(6) / sqrt(n + n_hidden))
    
    w1 = np.mat(np.random.rand(n_hidden, n_output))
    w1 = w1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - \
         np.mat(np.ones((n_hidden, n_output))) * \
         (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    
    b1 = np.mat(np.random.rand(1, n_output))
    b1 = b1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - \
         np.mat(np.ones((1, n_output))) * \
         (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    
    # 训练
    i = 0
    while i <= maxCycle:
        # 前向传播计算
        hidden_input = hidden_in(feature, w0, b0)  # a(2)
        hidden_output = hidden_out(hidden_input)   # z(2)
        output_in = predict_in(hidden_output, w1, b1)  # a(3)
        output_out = predict_out(output_in)            # z(3)
        
        # 反向传播
        # 输出层的残差
        delta_output = -np.multiply((label - output_out), partial_sig(output_in))
        # 中间层的残差: wji的值，需要对wij转秩获得
        delta_hidden = np.multiply((delta_output * w1.T), partial_sig(hidden_input))
        
        # 更行权重
        w1 = w1 - alpha * (hidden_output.T * delta_output)
        # 更新偏置？？？
        b1 = b1 - alpha * np.sum(delta_output, axis=0) * (1.0 / m)
        
        w0 = w0 - alpha * (feature.T * delta_hidden)
        b0 = b0 - alpha * np.sum(delta_hidden, axis=0) * (1.0 / m)
        
        if i % 100 == 0:
            costValue = (1.0/2) * get_cost(get_predict(feature, w0, w1, b0, b1)-label)
            print("iter: ", i, ", cost: ", costValue)                
        i += 1           
    return w0, w1, b0, b1


#-------运行区--------------------------------------------------------
if __name__ == '__main__':
    
    test_id = 0
    
    if test_id ==0:
        filename = 'data.txt'
        data, label, n_class = load_data(filename)
        
        label_value = []
        label = np.array(label)
        data = np.array(data)
        for value in np.array(label):
            label_value.append(value[0]*0 + value[1])            
        plt.scatter(data[:,0], data[:,1], c =['b' if v==0 else 'r' for v in label_value])
        
        # 训练模型，隐藏层节点数=20，最大迭代次数=1000， 学习率0.1，输出层节点数=标签种类
        w0, w1, b0, b1 = bp_train(data, label, 20, 1000, 0.1, n_class)
        
    elif test_id == 1:  # 用于预测一个新数据集
    
        w0, w1, b0, b1 = bp_train(data, label, 20, 1000, 0.1, n_class)
        get_predict(data, w0, w1, b0, b1)
        
        
    elif test_id == 2:  # 试试看能不能用在MNIST这种偏大的数据集上
        pass
        
    else:
        print('Wrong test_id!') 
        
    
    