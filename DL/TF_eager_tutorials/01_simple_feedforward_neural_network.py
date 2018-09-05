#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:57:53 2018

@author: suliang
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe  # 安装eager: pip3 tf-nightly

from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

tfe.enable_eager_execution()  # 启动eager execution

X, y = make_moons(n_samples=100, noise=0.1, random_state=2018)
# 利用sklearn生成数据[x1, x2], label = y, 为2分类问题
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.autumn)


# 定义一个类： 用于实现一个神经网络，在tensorflow中最容易的构造神经网络的方法是采用类
# 
class simple_nn(tf.keras.Model):
    def __init__(self):
        super(simple_nn, self).__init__()
        """ 在init里边定义dense layer全连接层，output layer输出层
        """   
        # Hidden layer.
        self.dense_layer = tf.layers.Dense(10, activation=tf.nn.relu)
        # Output layer. No activation.
        self.output_layer = tf.layers.Dense(2, activation=None)
    
    def predict(self, input_data):
        """ Runs a forward-pass through the network.     
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).   
            Returns:
                logits: unnormalized predictions.
        """
        hidden_activations = self.dense_layer(input_data)
        logits = self.output_layer(hidden_activations)
        return logits
    
    def loss_fn(self, input_data, target):  
        """ 定义损失函数
            采用tf自带的softmax交叉熵函数
        """
        logits = self.predict(input_data)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=logits)
        return loss
    
    def grads_fn(self, input_data, target):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)
        return tape.gradient(loss, self.variables)
    
    def fit(self, input_data, target, optimizer, num_epochs=500, verbose=50):
        """ Function to train the model, using the selected optimizer and
            for the desired number of epochs.
        """
        for i in range(num_epochs):
            grads = self.grads_fn(input_data, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if (i==0) | ((i+1)%verbose==0):
                print('Loss at epoch %d: %f' %(i+1, self.loss_fn(input_data, target).numpy()))
                
                
# 采用backpropagation反向传播的方式训练模型变量              
X_tensor = tf.constant(X)
y_tensor = tf.constant(y)

optimizer = tf.train.GradientDescentOptimizer(5e-1)
model = simple_nn()
model.fit(X_tensor, y_tensor, optimizer, num_epochs=500, verbose=50)




