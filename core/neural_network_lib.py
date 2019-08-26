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
from core.loss_function_lib import CrossEntropy, SquareLoss
from core.optimizer_lib import SGD, Adam, SGDM

from utils.dataloader import batch_iterator, train_test_split
from utils.matrix_operation import img2col, col2img
from dataset.digits_dataset import DigitsDataset
from dataset.regression_dataset import RegressionDataset
import matplotlib.pyplot as plt

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
        acc = self.loss_function.acc(y, y_pred)
        # 反向传播
        loss_grad = self.loss_function.gradient(y, y_pred)
        self.backward_pass(grad = loss_grad)
        return loss, acc
    
    def batch_operation_val(self, x, y):
        """在每个batch做完都做一次验证，比较费时，但可实时看到验证集的acc变化：可用于评估是否过拟合"""
        pass
    
    def train(self):
        total_iter = 1
        all_losses = []
        all_accs = []
        for i in range(self.n_epochs):
            it = 1
            for x_batch, y_batch in batch_iterator(self.feats, self.labels, 
                                                   batch_size = self.batch_size):
                loss, acc = self.batch_operation(x_batch, y_batch)
                # 显示loss
                if it % 5 == 0:
                    print("iter %d / epoch %d: loss=%f"%(it, i+1, loss))
                all_losses.append([total_iter,loss])
                all_accs.append([total_iter, acc])
                it += 1
                total_iter += 1
                
        self.vis_loss(all_losses, all_accs)
        self.trained = True  # 完成training则可以预测
        return all_losses
    
    def summary(self):
        """用来显示模型结构"""
        pass
    
    def evaluation(self, x, y):
        """对一组数据进行预测精度"""
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
        
        self.add(Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=1))# w'=(8-3+2)/1 +1=8
        self.add(Activation('relu'))
        self.add(BatchNorm2d())
        self.add(Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=1))
        self.add(Activation('relu'))
        self.add(BatchNorm2d())
        self.add(Flatten())
        self.add(Linear(in_features=2048, out_features=256))
        self.add(Activation('relu'))
        self.add(BatchNorm2d())
        self.add(Linear(in_features=256, out_features=10))
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


class LinearRegression(NeuralNetwork):
    """回归模型"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(feats=feats, labels=labels, 
                       loss=loss, 
                       optimizer=optimizer, 
                       n_epochs=n_epochs, 
                       batch_size=batch_size)
        self.add(Linear(in_features=1, out_features=1))
        
    def batch_operation(self, x, y):
        """基于每一个batch的数据分别进行前向计算和反向计算"""
        # 前向计算
        y_pred = self.forward_pass(x)  # 计算每一层的输出，这里就是linear层输出 (64,1)
        # 生成正则项的输入: 这里正则化的处理是统一加在loss端的梯度上
#        w_array = np.array([self.layers[0].W.item(), self.layers[0].W0.item()])  # W(1,1) & W0(1,1) -> (2,)
        losses = self.loss_function.loss(y.reshape(-1, 1), y_pred)
        loss = np.mean(losses)
        # 反向传播
        loss_grad = self.loss_function.gradient(y, y_pred)
        self.backward_pass(grad = loss_grad)
        return loss    
    
    def evaluation(self, x, y, title=None):
        """回归的评估：没有acc可评估，直接绘制拟合曲线"""
        y_pred = self.forward_pass(x)
        plt.figure()
        if title is None:
            title = 'regression curve'
        plt.title(title)
        plt.scatter(x, y, label='raw data', color='red')
        plt.plot(x, y_pred, label = 'y_pred', color='green')
        plt.legend()
        plt.grid()        


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
        self.W = np.random.uniform(-limit, limit, (self.in_features, self.out_features)) # ()
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
    """卷积层"""
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
        filter_h, filter_w = self.kernel_size
        out_h = (x.shape[2] + 2*self.padding - filter_h)//self.stride + 1  # 计算输出图像实际尺寸
        out_w = (x.shape[3] + 2*self.padding - filter_w)//self.stride + 1
        batch_size = x.shape[0]
        # 特征数据从matrix(256,1,8,8)变为列数据(16384, 9)
        self.x_col = img2col(x, filter_h, filter_w, self.stride, self.padding) # (9，16384)表示每个滤波器要跟图像进行的滤波次数为16384，所以把每组9元素都准备出来。
        self.w_col = self.W.reshape(-1, filter_h * filter_w)  # (16, 9)
        output = np.dot(self.w_col, self.x_col) + self.W0  #(16,9)*(9,16384)+(16,1) -> (16,16384) 列形式的w点积列形式的特征x
        output = output.reshape(self.out_channels,  out_h, out_w, batch_size)  # (16,8,8,256)
        # Redistribute axises so that batch size comes first
        return output.transpose(3,0,1,2)  #(256,16,8,8)
    
    def backward_pass(self, accum_grad):
        # 先获得反传梯度的分支到w
        grad_w = np.dot(accum_grad, self.W)  # ()
        grad_w0 = np.sum(accum_grad, axis=0)
        # 更新w
        self.W = self.W_optimizer.update(self.W, grad_w)
        self.W0 = self.W0_optimizer.update(self.W0, grad_w0)
        # 继续反传
        accum_grad = accum_grad * self.W
        # col形式转换回matrix
        accum_grad = col2img(accum_grad, )
        
        return accum_grad

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
    """归一化层"""
    def __init__(self):
        pass
    
    def forward_pass(self, x):
        pass
    
    def backward_pass(self, accum_grad):
        pass
    
            

class Flatten(Layer):
    """展平层"""
    def __init__(self):
        self.prev_shape = None
    
    def forward_pass(self, x):
        self.prev_shape = x.shape  # 预留输入的维度信息为反传做准备
        return x.reshape(x.shape[0], -1)
    
    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape) 


class Dropout(Layer):
    """关闭神经元层"""
    def __init__(self):
        pass
    
    def forward_pass(self, x):
        pass
    
    def backward_pass(self, accum_grad):
        pass


class MaxPooling2d(Layer):
    """最大池化层"""
    def __init__(self):
        pass
    
    def forward_pass(self, x):
        pass
    
    def backward_pass(self, accum_grad):
        pass
    

class AvgPooling2d(Layer):
    """平均池化层"""
    def __init__(self):
        pass
    
    def forward_pass(self, x):
        pass
    
    def backward_pass(self, accum_grad):
        pass


# %% 调试
if __name__ == "__main__":
    
    model = 'mlp'
    
    if model == 'softmax':  # 输入图片是(b,n)
    
        dataset = DigitsDataset(norm=True, one_hot=True)
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels, test_size=0.3, shuffle=True)
        
        optimizer = SGDM(lr=0.001)
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
        dataset = DigitsDataset(norm=True, one_hot=True)
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels, test_size=0.3, shuffle=True)
        
        # 已展平数据转matrix
        train_x = train_x.reshape(-1, 1, 8, 8)
        test_x = test_x.reshape(-1, 1, 8, 8)
        
        optimizer = Adam(lr=0.001)
        loss_func = CrossEntropy()
        clf = CNN(train_x, train_y, 
                  loss=loss_func, 
                  optimizer= optimizer, 
                  batch_size = 64, 
                  n_epochs=200)
        clf.train()
        acc1, _ = clf.evaluation(train_x, train_y)
        print("training acc: %f"%acc1)
        acc2, _ = clf.evaluation(test_x, test_y)
        print("test acc: %f"%acc2)
        
    if model == 'reg':
        dataset = RegressionDataset(n_samples=500, n_features=1, n_targets=1, noise=4)
        X = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)    
        
        train_x = train_x.reshape(-1, 1)
        test_x = test_x.reshape(-1, 1)
        train_y = train_y.reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)
        
        optimizer = SGD(lr=0.0001, weight_decay=0.1, regularization_type='l2')  # TODO: 这里暂时只有SGD/SGDM能够收敛，Adam不能
        loss_func = SquareLoss()
#        regularization = no_regularization()
#        regularization = l2_regularization(weight_decay=0.1)
        reg = LinearRegression(train_x, train_y, 
                               loss=loss_func, 
                               optimizer= optimizer, 
                               batch_size = 64, 
                               n_epochs=500)
        reg.train()
        reg.evaluation(train_x, train_y, "train")
        reg.evaluation(test_x, test_y, "test")        
        
        
        
        