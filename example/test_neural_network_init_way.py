#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:20:23 2019

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt

from core.neural_network_lib import NeuralNetwork, Linear, Activation 
from utils.dataloader import train_test_split
from dataset.digits_dataset import DigitsDataset
from core.loss_function_lib import CrossEntropy
from core.optimizer_lib import SGD, SGDM
"""
权重初始化的最佳实践：
1. 必须随机初始化权重，防止权重均一化：
2. 必须尽可能小的初始化权重，防止权重值过大造成过拟合：
3. 必须
"""


class MLP5L(NeuralNetwork):
    """基于神经网络结构的多层卷积神经网络"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size, 
                 val_feats=None, val_labels=None):
        
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size,
                         val_feats=val_feats,
                         val_labels=val_labels)
        
        self.add(Linear(in_features=64, out_features=100))
        self.add(Activation('relu'))
        self.add(Linear(in_features=100, out_features=100))
        self.add(Activation('relu'))
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
        dataset = DigitsDataset(norm=True, one_hot=True)
        # 定义一个过拟合的数据：即减少数据量到300个数
        datas = dataset.datas[:200]
        labels = dataset.labels[:200]
        # 分隔数据集
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, 
                                                            dataset.labels, 
                                                            test_size=0.3, 
                                                            shuffle=True)
        test_x, val_x, test_y, val_y = train_test_split(test_x, 
                                                        test_y, 
                                                        test_size=0.3, 
                                                        shuffle=True)
        # 尝试不带正则化的SGDM和带正则化的SGDM的区别
        optimizer = SGDM(lr=0.001, regularization_type=None)
#        optimizer = SGDM(lr=0.001, regularization_type='l2')
        
        loss_func = CrossEntropy()
        clf = MLP5L(train_x, train_y, 
                    loss=loss_func, 
                    optimizer= optimizer, 
                    batch_size = 100, 
                    n_epochs=100,
                    val_feats=val_x,
                    val_labels=val_y)
        _, all_accs = clf.train()
        acc1, _ = clf.evaluation(train_x, train_y)
        print("training acc: %f"%acc1)
        acc2, _ = clf.evaluation(test_x, test_y)
        print("test acc: %f"%acc2)
        
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
        