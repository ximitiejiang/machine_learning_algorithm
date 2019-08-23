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
from core.activation_function_lib import Relu, LeakyRelu, Elu, Sigmoid, Softmax
from core.loss_function_lib import CrossEntropy
from core.optimizer_lib import Adam, SGDM

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
        model_type: sl(shallow learning/supervised learning) or dl(deep learning)
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
#        if self.layers:  # 如果不为空，说明不是第一层，则取前一层为该层输入
#            layer.set_input_shape(shape=self.layers[-1].get_output_shape())
        
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
            
    def batch_operation(self, x, y):
        """基于每一个batch的数据分别进行前向计算和反向计算"""
        # 前向计算
        y_pred = self.forward_pass(x)
        losses = self.loss_function.loss(y, y_pred)
        loss = np.mean(losses)
        # 反向传播
        loss_grad = self.loss_function.gradient(y, y_pred)
        self.backward_pass(grad = loss_grad)
        return loss
    
    def train(self):
        total_iter = 1
        all_losses = []
        for i in range(self.n_epochs):
            it = 1
            for x_batch, y_batch in batch_iterator(self.feats, self.labels, 
                                                   batch_size = self.batch_size):
                loss = self.batch_operation(x_batch, y_batch)
                # 显示loss
                if it % 5 == 0:
                    print("iter %d / epoch %d: loss=%f"%(it, i+1, loss))
                all_losses.append([total_iter,loss])
                it += 1
                total_iter += 1
                
        self.vis_loss(all_losses)
        self.trained = True  # 完成training则可以预测
        return all_losses
    
    def summary(self):
        """用来显示模型结构"""
        pass
    
    def evaluation(self, x, y):
        """对一组数据进行精度"""
        if self.trained:
            y_pred = self.forward_pass(x)  # (1280, 10)
            y_pred = np.argmax(y_pred, axis=1)  # (1280,)
            y_label = np.argmax(y, axis=1)      # (1280,)
            acc = np.sum(y_pred == y_label, axis=0) / len(y_label)
            return acc, y_pred
        else:
            raise ValueError("model not trained, can not predict or evaluate.")
    
     
    
class CNN(NeuralNetwork):
    """基于神经网络结构的多层卷积神经网络"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
        
        self.add(Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding='same'))
        self.add(Activation('relu'))
        self.add(BatchNorm2d())
        self.add(Conv2d(in_channels=1, out_channels=32, kernel_sie=(3,3), stride=1, padding='same'))
        self.add(Activation('relu'))
        self.add(BatchNorm2d())
        self.add(Flatten())
        self.add(Linear(output_shape=(256,)))
        self.add(Activation('relu'))
        self.add(BatchNorm2d())
        self.add(Linear(output_shpae=(10,)))
        self.add(Activation('softmax'))
        
    
class MLP(NeuralNetwork):
    """基于神经网络结构的多层感知机"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
        n_feats = feats.shape[1]
        n_classes = labels.shape[1]
        self.add(Linear(in_features=n_feats, out_features=16))
        self.add(Activation('elu'))
        self.add(Linear(in_features=16, out_features=n_classes))
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
        self.add(Linear(in_features=n_feats, out_features=n_classes))
        self.add(Activation('softmax'))
        

# %% 单层模型
class Layer():
    """层的基类: 该基类不需要初始化，用于强制定义需要实施的方法，并把所有共有函数放在一起"""
    def forward_pass(self, x):
        raise NotImplementedError()
    
    def backward_pass(self, accum_grad):
        raise NotImplementedError()
        
    
class Linear(Layer):
    """全连接层: 要求输入必须是经过flatten的二维数据(n_samples, n_feats)"""
    def __init__(self, in_features, out_features):  
        self.out_features = out_features  # (m,)
        self.in_features = in_features    # (n,)
        self.trainable = True
        self.W = None
        self.W0 = None
    
    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.in_features)
        self.W = np.random.uniform(-limit, limit, (self.in_features, self.out_features)) # w()
        self.W0 = np.zeros((1, self.out_features))  # TODO  w0(1,10)
        self.W_optimizer = copy.copy(optimizer)
        self.W0_optimizer = copy.copy(optimizer)
    
    def forward_pass(self, x):
        self.input_feats = x     # 保存输入，为反传预留
        return np.dot(x, self.W) + self.W0  # (8,64)(64,10)+(1,10) ->(8,10)+(1,10)
    
    def backward_pass(self, accum_grad): 
        tmp_W = self.W
        # 更新参数
        if self.trainable:
            grad_w = np.dot(self.input_feats.T, accum_grad)  # 关键1:计算权值梯度时是对w求导得到的是x所以是累积梯度乘以输入x，(8,64).T (8,10) -> (64,10)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)   # (64,10)->(1,10)  TODO: 偏置参数的更新逻辑
            
            self.W = self.W_optimizer.update(self.W, grad_w)      
            self.W0 = self.W0_optimizer.update(self.W0, grad_w0)
        # 累积梯度
        accum_grad = np.dot(accum_grad, tmp_W.T)           # 关键2:计算累积梯度时是对x求导得到的是w所以是累积梯度乘以权值w  
        return accum_grad
    
        
class Conv2d(Layer):
    """卷积层: 当前仅支持stride=1"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def initialize(self, optimizer):
        h, w = self.kernel_size
        
        limit = 1 / math.sqrt(np.prod(self.kernel_size))
        # 初始化卷积参数w和偏置w0
        self.W = np.random.uniform(-limit, limit, size=(self.out_channels ,self.in_channels, h, w))  # (16,1,3,3)
        self.W0 = np.zeros((self.out_channels, 1)) #(16,1)
        self.W_optimizer = copy.copy(optimizer)
        self.W0_optimizer = copy.copy(optimizer)
    
    def forward_pass(self, x):
        self.x_col = img2col()
        self.w_col
        
        output = np.dot(self.w_col, self.x_col) + self.W0  #(16,9)*(9,16384)+(16,1) -> (16,16384) 列形式的w点积列形式的特征x
        output = output.reshape()
        
        return output
    
    def backward_pass(self, grad):
        pass
    
    def img2col(self, img, kernel_size, ):
        """用于把图片数据转换成列数据
        """
        img_h = img.shape[2]  # 获得原图h,w
        img_w = img.shape[2]
        kernel_h, kernel_w = kernel_size.shape
        pad_h = (img_h - kernel_h) + 
        pad_w = 
        padded = np.pad(img, ((0,0), (0,0)), mode="constant")  # (256,1,8,8)->(256,1,10,10)
        
        i, j, k = get_img2col_inds()
        
        cols = padded[:, k, i, j]    # (256,1,10,10) -> (256, 9, 64)这是把图片数据
        cols = cols.transpose(1,2,0).reshape()  # (9,16384)
        return cols
        
        
        
    def col2img(self):   
        """用于把列数据转换回图片数据，col2img跟img2col互为逆运算"""
        
        
        

class Activation(Layer):
    """激活层: 基于输入的激活函数类型name来生成激活函数对象进行调用"""
    activation_dict = {'relu': Relu,
                       'sigmoid': Sigmoid,
                       'softmax': Softmax,
                       'leakyrelu': LeakyRelu,
                       'elu': Elu}
    def __init__(self, name):
        self.name = name
        self.activation_func = self.activation_dict[name]()  # 创建激活函数对象
        self.trainable = True
    
    def forward_pass(self, x):
        self.input_feats = x
        return self.activation_func(x)
    
    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.input_feats)
    
        
    
class BatchNorm2d(Layer):
    
    def __init__(self):
        pass
    
    def forward_pass(self):
        pass
    
    def backward_pass(self):
        pass
    
            

class Flatten(Layer):
    
    def __init__():
        pass
    
    def forward_pass(self):
        pass
    
    def backward_pass(self):
        pass
    

# %% 调试
if __name__ == "__main__":
    
    model = 'mlp'
    
    if model == 'logi':  # 输入图片是(b,n)
    
        dataset = DigitsDataset(norm=True, one_hot=True)
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels, test_size=0.3, shuffle=True)
        
        optimizer = Adam(lr=0.001)
        loss_func = CrossEntropy()
        clf = SoftmaxReg(train_x, train_y, 
                         loss=loss_func, 
                         optimizer= optimizer, 
                         batch_size = 64, 
                         n_epochs=200)
        clf.train()
        acc1, _ = clf.evaluation(train_x, train_y)
        print("training acc: %f"%acc1)
        acc2, _ = clf.evaluation(test_x, test_y)
        print("test acc: %f"%acc2)
           
    if model == "mlp":  # 输入图片是(b,n)
        dataset = DigitsDataset(norm=True, one_hot=True)
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels, test_size=0.3, shuffle=True)
        
        optimizer = SGDM(lr=0.001, momentum=0.9)
        loss_func = CrossEntropy()
        clf = MLP(train_x, train_y, 
                  loss=loss_func, 
                  optimizer= optimizer, 
                  batch_size = 64, 
                  n_epochs=200)
        clf.train()
        acc1, _ = clf.evaluation(train_x, train_y)
        print("training acc: %f"%acc1)
        acc2, _ = clf.evaluation(test_x, test_y)
        print("test acc: %f"%acc2)
    
    if model == "cnn":  # 输入图片必须是(b,c,h,w)
        pass
    
    
    