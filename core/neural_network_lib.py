#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:22:32 2019

mainly refer to https://github.com/eriklindernoren/ML-From-Scratch

@author: ubuntu
"""

import numpy as np
import math
import copy
from core.base_model import BaseModel
from utils.dataloader import batch_iterator, train_test_split
from dataset.digits_dataset import DigitsDataset

# %%  模型总成
class NeuralNetwork(BaseModel):
    """神经网络模型：这个NN是一个空壳类，并不包含任何层，用于模型创建的基类。
    Args:
        feats: (n_sample, n_feats) 
        labels: (n_sample, n_classes) 独热编码形式
        loss: 传入loss对象，该对象实现了__call__方法可直接调用
        optimizer: 传入optimizer对象，该对象主要是采用update(w, grad)方法
    """
    def __init__(self, feats, labels, 
                 loss, optimizer, n_epochs, batch_size):
        super().__init__(feats, labels)
        
        self.optimizer = optimizer
        self.loss_function = loss
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.layers = []

    def add(self, layer):
        """用于模型添加层: 设置输入输出层数，初始化"""
        if self.layers:  # 如果不为空，说明不是第一层，则取前一层为该层输入
            layer.set_input_shape(shape=self.layers[-1].get_output_shape())
        
        if hasattr(layer, "initialize"):
            layer.initialize(optimizer = self.optimizer)
        
        self.layers.append(layer)
    
    def forward_pass(self, x):
        """前向计算每一层的输出"""
        for layer in self.layers:
            layer_output = layer.forward_pass(x)
            x = layer_output
        return layer_output
    
    def backward_pass(self, grad):
        """基于梯度反向更新每一层的累积梯度同时更新每一层的梯度"""
        for layer in self.layers[::-1]:
            grad = layer.backward_pass(grad)  # 梯度的反向传播，且传播的是每一层的累积梯度
            
    def train_batch_op(self, x, y):
        """基于每一个batch的数据分别进行前向计算和反向计算"""
        y_pred = self.forward_pass(x)
        losses = self.loss_function.loss(y, y_pred)
        loss = np.mean(losses)
        loss_grad = self.loss_function.gradient(y, y_pred)
        
        self.backward_pass(grad = loss_grad)
        
        return loss
    
    def train(self):
        for i in range(self.n_epochs):
            
            all_losses = []
            it = 1
            for x_batch, y_batch in batch_iterator(self.feats, self.labels, batch_size = self.batch_size):
                
                loss = self.train_batch_op(x_batch, y_batch)
                # 显示loss
                if it % 20 == 0:
                    print("iter %d / epoch %d: loss=%f"%(it, i, loss))
                all_losses.append(loss)
                it += 1
                
        self.vis_loss(all_losses)
        return all_losses
    
    def test_batch_op(self):
        pass
    
    def summary(self):
        pass
    
    def evaluation(self):
        """对整个数据集进行评估"""
    
    def predict(self, X):
        """对一组数据进行预测"""
        pass
    
        
    
class CNN(NeuralNetwork):
    """基于神经网络结构的多层卷积神经网络"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(self, feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
        
        self.add(Conv2D())
        
    
class MLP(NeuralNetwork):
    """基于神经网络结构的多层感知机"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(self, feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
        n_feats = feats.shape[1]
        n_classes = labels.shape[1]
        self.add(Linear(input_shape=(n_feats,), output_shape=(16,)))
        self.add(Activation('relu'))
        self.add(Linear(output_shape=(n_classes,)))
        self.add(Activation('softmax'))


class SoftmaxReg(NeuralNetwork):
    """基于神经网络结构的多分类softmax模型"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
        n_feats = feats.shape[1]
        n_classes = labels.shape[1]
        self.add(Linear(input_shape=(n_feats,), output_shape=(n_classes,)))
        self.add(Activation('softmax'))
        

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
        return x if x>=0 else 0
    
    def gradient(self, x):
        pass
        

# %% 单层模型
class Layer():
    """层的基类: 该基类不需要初始化，用于强制定义需要实施的方法，并把所有共有函数放在一起"""
    def set_input_shape(self, shape):  # 表示单样本形状：如果是卷积层输入(h, w), 如果是全连接层输入(n_feats,) 
        self.input_shape = shape
    
    def get_output_shape(self):
        raise NotImplementedError()
    
    def forward_pass(self, x):
        raise NotImplementedError()
    
    def backward_pass(self, accum_grad):
        raise NotImplementedError()
        
    
class Linear(Layer):
    """全连接层: 要求输入必须是经过flatten的二维数据(n_samples, n_feats)"""
    def __init__(self, output_shape, input_shape=None):  # 作为中间层，只需要输入output_channels(value)，而作为首层，则需同时输入output_channel和input_shape(b,c), 
        self.output_shape = output_shape  # (m,)
        self.input_shape = input_shape    # (n,)
        self.trainable = True
        self.W = None
        self.W0 = None
    
    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.output_shape[0])) # w()
        self.W0 = np.zeros((1, self.output_shape[0]))  # TODO  w0(1,10)
        self.W_optimizer = copy.copy(optimizer)
        self.W0_optimizer = copy.copy(optimizer)
    
    def forward_pass(self, x):
        self.input_feats = x
        return np.dot(x, self.W) + self.W0  # (8,64)(64,10)+(1,10) ->(8,10)+(1,10)
    
    def backward_pass(self, accum_grad): 
        tmp_W = self.W
        # 更新参数
        if self.trainable:
            grad_w = np.dot(self.input_feats.T, accum_grad)  # (8,64).T (8,10) -> (64,10)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)   # (64,10)->(1,10)  TODO: 偏置参数的更新逻辑
            
            self.W = self.W_optimizer.update(self.W, grad_w)
            self.W0 = self.W0_optimizer.update(self.W0, grad_w0)
        # 累积梯度
        accum_grad = np.dot(accum_grad, tmp_W.T)  # TODO： 梯度反传时，全连接层的梯度不是x吗？怎么是乘以w了？
        return accum_grad
    
    def get_output_shape(self):
        return self.output_shape
        
        
class Conv2D(Layer):
    """卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def init_parameters(self):
        # 初始化卷积参数w
        self.w = np.random.uniform()
        # 初始化偏置参数b
        self.w0 = np.zeros()
    
    def forward_pass(self, x):
        pass
    
    def backward_pass(self, grad):
        pass
    
    def calc_output_shape(self):
        """实时计算层的输出数据结构: w' = (w - k_size + 2*pad) / stride +1"""

        return
    
    def calc_input_shape(self):
        """实时计算层的输入数据结构"""


class Activation(Layer):
    """激活层: 基于输入的激活函数类型name来生成激活函数对象进行调用"""
    activation_dict = {'relu': Relu,
                       'sigmoid': Sigmoid,
                       'softmax': Softmax}
    def __init__(self, name):
        self.name = name
        self.activation_func = self.activation_dict[name]()  # 创建激活函数对象
        self.trainable = True
    
    def forward_pass(self, x):
        self.input_feats = x
        return self.activation_func(x)
    
    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.input_feats)
    
    def output_shape(self):
        return self.input_shape
        
        

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
        return w - self.w_update


class SGD():
    """普通SGD梯度下降算法"""
    def __init__(self, lr=0.001):
        self.lr = lr
    
    def update(self, w, grad):
        w -= self.lr * grad
        return w

class SGDM():
    """SGDM梯度下降算法-带动量M的SGD算法"""
    def __init__(self, lr, momentum=0.9, weight_decay=0.0001):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.m = 0  # 前一次的
    
    def update(self, w, grad):
        self.m = self.momentum * self.m + self.weight_decay  # 计算更新m
        w -= self.lr * self.m
        return 
        

    
    
# %% 调试
if __name__ == "__main__":
    
    model = 'logi'
    
    if model == 'logi':
    
        dataset = DigitsDataset(norm=True, one_hot=True)
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels)
        
        optimizer = Adam(lr=0.001)
        loss_func = CrossEntropy()
        clf = SoftmaxReg(train_x, train_y, 
                         loss=loss_func, 
                         optimizer= optimizer, 
                         batch_size = 64, 
                         n_epochs=500)
        clf.train()
#        acc1 = clf.evaluation(train_x, train_y)
#        print("training acc: %f"%acc1)
#        
#        acc2 = clf.evaluation(test_x, test_y)
#        print("test acc: %f"%acc2)
        
    
    
    
    
    