#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 18:04:49 2019

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt

from core.neural_network_lib import NeuralNetwork, Linear, Activation, Conv2d, BatchNorm2d, Flatten, MaxPool2d
from utils.dataloader import train_test_split
from dataset.digits_dataset import DigitsDataset
from core.loss_function_lib import CrossEntropy
from core.optimizer_lib import SGD, SGDM, RMSprop, Adam


class CNN(NeuralNetwork):
    """基于神经网络结构的多层卷积神经网络"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size,
                 val_feats=None, val_labels=None):
        """本模型为4层神经网络(2层卷积+2层全连接)，基于输入图形必须为gray(1,8,8)(c,h,w), batchsize可自定义。
        """
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size,
                         val_feats=val_feats,
                         val_labels=val_labels)
        afn = 'relu'  # 该模型下elu得到精度最高96.88%
        self.add(Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=1))# w'=(8-3+2)/1 +1=8
        self.add(BatchNorm2d(16))
        self.add(Activation(afn))

#        self.add(Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=1))# w'=(8-3+2)/1 +1=8
#        self.add(Activation(afn))
#        self.add(BatchNorm2d(32))
        self.add(Flatten())  # (64,16,8,8) -> (64,16*8*8=1024)
        self.add(Linear(in_features=1024, out_features=256))  #
        self.add(BatchNorm2d(256))
        self.add(Activation(afn))
        self.add(Linear(in_features=256, out_features=10))
        self.add(BatchNorm2d(256))
        self.add(Activation('softmax'))
    
    def forward_pass(self, x, training):
        """重写前向计算，提取激活值"""
        self.activations = []
        self.linears = []
        self.bns = []
        for i, layer in enumerate(self.layers):
            layer_output = layer.forward_pass(x, training=training)
            x = layer_output
            
            if i in [0,4,7] : # linear层的激活值输出
                self.linears.append(layer_output)
            if i in [1,5,8] : # batchNorm输出
                self.bns.append(layer_output)
            if i in [2,6,9] : # activation输出
                self.activations.append(layer_output)                
        
        return layer_output


class CNN1(NeuralNetwork):
    """1层卷积神经网络"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size,
                 val_feats=None, val_labels=None):
        """本模型为4层神经网络(2层卷积+2层全连接)，基于输入图形必须为gray(1,8,8)(c,h,w), batchsize可自定义。
        """
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size,
                         val_feats=val_feats,
                         val_labels=val_labels)
        # w' = (4-3+0)/1+1=2
        self.add(Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=0))
        self.add(Flatten())
        self.add(Linear(in_features=4, out_features=2))
        self.add(Activation('softmax'))


class CNNMax(NeuralNetwork):
    """测试池化层"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size,
             val_feats=None, val_labels=None):
        """本模型为4层神经网络(2层卷积+2层全连接)，基于输入图形必须为gray(1,8,8)(c,h,w), batchsize可自定义。
        """
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size,
                         val_feats=val_feats,
                         val_labels=val_labels)
        # w' = (4-3+0)/1+1=2
        self.add(Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=0))
        self.add(Flatten())
        self.add(Linear(in_features=4, out_features=2))
        self.add(Activation('softmax'))
        self.add(MaxPool2d(kernel_size=(3,3), stride=1, padding=0))
        self.add()

    
    
if __name__ == "__main__":
    
    id='maxpool'
    
    if id=='cnn1':
        img = np.array([1,2,3,4, 2.,0,1,3, 3,1,0,2, 4,2,1,0]).reshape(1,1,4,4) 
        label = np.array([[0., 1.]])
        optimizer = SGDM(lr=0.01, 
                         weight_decay=0.1, regularization_type='l2')
        loss_func = CrossEntropy()
        clf = CNN1(img, label, 
                  loss=loss_func, 
                  optimizer= optimizer, 
                  batch_size = 1, 
                  n_epochs=1)
        clf.train()
        clf.evaluation(img, label, "train")
    
    
    if id == "maxpool":
        img = np.array([1,2,3,4, 2.,0,1,3, 3,1,0,2, 4,2,1,0]).reshape(1,1,4,4) 
        label = np.array([[0., 1.]])
        optimizer = SGDM(lr=0.01, 
                         weight_decay=0.1, regularization_type='l2')
        loss_func = CrossEntropy()
        clf = CNNMax(img, label, 
                     loss=loss_func, 
                     optimizer= optimizer, 
                     batch_size = 1, 
                     n_epochs=1)
        clf.train()
    
    
    if id=='cnn':
        
        dataset = DigitsDataset(norm=True, one_hot=True)

        datas = dataset.datas
        labels = dataset.labels
        # 分隔数据集
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, 
                                                            dataset.labels, 
                                                            test_size=0.3, 
                                                            shuffle=True)
#        test_x, val_x, test_y, val_y = train_test_split(test_x, 
#                                                        test_y, 
#                                                        test_size=0.3, 
#                                                        shuffle=True)
        train_x = train_x.reshape(-1, 1, 8, 8)
        test_x = test_x.reshape(-1, 1, 8, 8)
#        val_x = val_x.reshape(-1, 1, 8, 8)
        
        optimizer = SGDM(lr=0.01, 
                         weight_decay=0.1, regularization_type='l2')
        
        loss_func = CrossEntropy()
        clf = CNN(train_x, train_y, 
                    loss=loss_func, 
                    optimizer= optimizer, 
                    batch_size = 128, 
                    n_epochs=10)
        _, accs = clf.train()
        test_batch_size=128
        acc1, _ = clf.evaluation(train_x[:test_batch_size], train_y[:test_batch_size], "train")
        acc2, _ = clf.evaluation(test_x[:test_batch_size], test_y[:test_batch_size], "test")

        linears = clf.linears  # (6,) (64, 100)
        # 绘制直方图统计各种激活值取值范围
        plt.figure()
        for i, li in enumerate(linears):
            plt.subplot(2, len(linears)/2+1, i+1)
            plt.title(str(i+1)+"-linear")
            plt.hist(li.flatten(), 30, range=(-3,3))  # 展平成(6400,), 然后取30个区间, 只统计取值在0-1之间的数
        plt.show()
        
        bns = clf.bns  # (6,) (64, 100)
        # 绘制直方图统计各种激活值取值范围
        plt.figure()
        for i, bn in enumerate(bns):
            plt.subplot(2, len(bns)/2+1, i+1)
            plt.title(str(i+1)+"-bn")
            plt.hist(bn.flatten(), 30, range=(-3,3))  # 展平成(6400,), 然后取30个区间, 只统计取值在0-1之间的数
        plt.show()
        
        
        activations = clf.activations  # (6,) (64, 100)
        # 绘制直方图统计各种激活值取值范围
        plt.figure()
        for i, ac in enumerate(activations):
            plt.subplot(2, len(activations)/2+1, i+1)
            plt.title(str(i+1)+"-activate")
            plt.hist(ac.flatten(), 30, range=(-3,3))  # 展平成(6400,), 然后取30个区间, 只统计取值在0-1之间的数
        plt.show()