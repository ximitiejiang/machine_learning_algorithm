#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:14:45 2019

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt

from core.neural_network_lib import NeuralNetwork, Linear, Activation, Conv2d, BatchNorm2d, Flatten, MaxPool2d
from utils.dataloader import train_test_split
from dataset.digits_dataset import DigitsDataset
from core.loss_function_lib import CrossEntropy
from core.optimizer_lib import SGD, SGDM, RMSprop, Adam


class LeNet5(NeuralNetwork):
    """深度卷积网络鼻祖LeNet5"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size,
                 val_feats=None, val_labels=None):
        """本模型为4层神经网络(2层卷积+2层全连接)，
        输入图形：(b,c,32,32)
        """
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size,
                         val_feats=val_feats,
                         val_labels=val_labels)
        afn = 'relu'  
        self.add(Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1, padding=1))# w'=(8-3+2)/1 +1=8
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


if __name__ == "__main__":
    
    id = 'lenet5'
    
    if id=='lenet5':
        
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
        clf = LeNet5(train_x, train_y, 
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
