#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:20:23 2019

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt

from core.neural_network_lib import NeuralNetwork, Linear, Activation, BatchNorm2d, Empty
from utils.dataloader import train_test_split
from dataset.digits_dataset import DigitsDataset
from core.loss_function_lib import CrossEntropy
from core.optimizer_lib import SGD, SGDM
"""
数据的变换过程：
    - 初始数据：图像特征已被标准化到N(0-1)的标准正态分布，取值在-2.x~+2.x之间
    - 经过全连接层：线性变换，数据分布形状基本不变，
    - 经过激活函数：数据分布开始改变并逐渐变成比较平衡有一定广度的分布，这样才能让数据保持多样性，否则相当于只剩下少量特征，其他特征被丢弃了。
      比如sigmoid会让数据的负值特征变成正值导致特征偏移(糟糕特性)。relu会只提取数据大于0的右半部分(负特征去除但特征没有移动)，

权重初始化的最佳实践：
1. 必须随机初始化权重，防止权重均一化：
2. 必须尽可能小的初始化权重，防止权重值过大造成过拟合：
3. 必须尽可能让激活层的输出保持一定的广度数据，才能保证模型学习到东西。
"""


class MLP7L(NeuralNetwork):
    """基于神经网络结构的多层全连接神经网络"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size, 
                 val_feats=None, val_labels=None):
        
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size,
                         val_feats=val_feats,
                         val_labels=val_labels)
        
        afn = 'relu'
        with_bn = True
        if with_bn:
            self.add(Linear(in_features=64, out_features=100))
            self.add(BatchNorm2d(100))
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(BatchNorm2d(100))
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(BatchNorm2d(100))
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(BatchNorm2d(100))
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(BatchNorm2d(100))
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(BatchNorm2d(100))
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=10))
            self.add(Activation('softmax'))
        else:
            self.add(Linear(in_features=64, out_features=100))
            self.add(Empty())
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(Empty())
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(Empty())
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(Empty())
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(Empty())
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=100))
            self.add(Empty())
            self.add(Activation(afn))
            self.add(Linear(in_features=100, out_features=10))
            self.add(Activation('softmax'))
    
    def forward_pass(self, x, training):
        """重写前向计算，提取激活值"""
        self.activations = []
        self.linears = []
        self.bns = []
        for i, layer in enumerate(self.layers):
            layer_output = layer.forward_pass(x, training=training)
            x = layer_output
            
            if i % 3 == 0 : # linear层的激活值输出
                self.linears.append(layer_output)
            if i % 3 == 1 : # batchNorm输出
                self.bns.append(layer_output)
            if i % 3 == 2 : # activation输出
                self.activations.append(layer_output)                
        
        return layer_output
    

        
if __name__ == "__main__":
        dataset = DigitsDataset(norm=True, one_hot=True)
        datas = dataset.datas
        labels = dataset.labels
        # 分隔数据集
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, 
                                                            dataset.labels, 
                                                            test_size=0.3, 
                                                            shuffle=True)
        test_x, val_x, test_y, val_y = train_test_split(test_x, 
                                                        test_y, 
                                                        test_size=0.3, 
                                                        shuffle=True)

        optimizer = SGDM(lr=0.001, regularization_type=None)
        loss_func = CrossEntropy()
        clf = MLP7L(train_x, train_y, 
                    loss=loss_func, 
                    optimizer= optimizer, 
                    batch_size = 64, 
                    n_epochs=20,
                    val_feats=val_x,
                    val_labels=val_y)
        _, all_accs = clf.train()
        acc1, _ = clf.evaluation(train_x, train_y, "train")
        acc2, _ = clf.evaluation(test_x, test_y, "test")

        
        # 比较train, val的acc
        train_acc = np.array(all_accs['train'])
        val_acc = np.array(all_accs['val'])
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('acc compare of train and val')
        ax.plot(train_acc[:, 0], train_acc[:, 1], label='train acc')
        ax.plot(val_acc[:, 0], val_acc[:, 1], label='val acc')        
        ax.legend()
        ax.grid()
        

        
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
        