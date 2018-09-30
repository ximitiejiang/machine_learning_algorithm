#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:57:53 2018

@author: suliang
xx.__version
    * tf: 1.11.0-dev20180904
    * np: 1.14.0
    * tfe
    
基于新安装的eager execution进行TF的实例调试练习
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
        # 隐藏层
        self.dense_layer = tf.layers.Dense(10, activation=tf.nn.relu)
        # 输出层. No activation.
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


# 可视化
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict target for each sample xx, yy
Z = np.argmax(model.predict(tf.constant(np.c_[xx.ravel(), yy.ravel()])).numpy(), axis=1)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.autumn, alpha=0.8)

# Plot our training points
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.autumn, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('First feature', fontsize=15)
plt.ylabel('Second feature', fontsize=15)
plt.title('Toy classification problem', fontsize=15)
plt.show()


