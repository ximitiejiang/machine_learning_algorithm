#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:30:36 2019

@author: ubuntu
"""
#%%
"""
该例程来自：https://github.com/czy36mengfei/tensorflow2_tutorials_chinese
"""
#%% 导入
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

#%%获得数据
"""keras的默认下载源是google，速度太慢无法下载，需要自己找其他源手动下载，然后放到指定目录
    1. 数据集：放到.keras/datasets/
        一般数据集只需要把gz文件放到该文件夹下面作为一级文件即可，
        但fashion-mnist数据集需要先新建fashion-mnist文件夹
        而cifar-10和cifar-100则除了需要下载cifar010外还需要下载cifar-10-batches-py.tar.gz
    2. 模型文件：放到.keras/models/
        基础预训练模型文件
    3. 数据集fashion mnist的基本信息
        - 数据量是70,000张图片, 图片大小28*28, gray单通道图片，包含10个类别的多分类问题
    4. 其他可导入的数据集
        keras.datasets.mnist.load_data()
        keras.datasets.fashion_mnist.load_data()
"""
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
# 总计10个分类，对应标签0-9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)  # 训练集: 60,000
print(train_labels.shape)
print(test_images.shape)   # 测试集: 10,000
print(test_labels.shape)

#%% 对数据做归一化
train_images = train_images / 255.0  #做简单的normalize归一化处理到[0-1]之间，也可以做standardlize标准化到N(0-1)
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):          # 这段plt的显示代码不错，方便拼接(不显示坐标轴，可显示下标题，可subplot拼接)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)  # 由于做了归一化，这里就需要设置plt.cm.binary
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#%% 构造模型和编译器
"""
    1. 创建模型model = tf.keras.Sequential()
    
    2. 添加层model.add()，可使用的层包括
        layers.Dense()
    注意：每层可以指定activation(激活函数), kernel_initializer(初始化权重), 
    kernel_regularizer(权重正则化), bias_regularizer(偏置正则化)
    其中，默认没有激活函数，可选择"relu", "softmax",...
         默认初始化方式为Glorot uniform，可选择tf.keras.initializers.glorot_normal
         默认不适用正则化函数，可选择tf.keras.regularizers.l2(0.01), tf.keras.regularizers.l1(0.01)
    
    3. 设置训练流程model.compile()
    需要至少定义optimizer, loss, metrics
    其中，optimizer可选tf.keras.optimizers.Adam(0.001)
         loss可选tf.keras.losses.categorical_crossentropy
         metrics可选[tf.keras.metrics.categorical_accuracy]

    4. 查看模型参数：model.summary()
    5. 查看模型结构图：先安装graphviz和pydot
        sudo apt-get install graphviz
        pip3 install pydot
        然后就可以：keras.utils.plot_model(model, "model_info.png", show_shapes=True)
"""
model = keras.Sequential(
[
    layers.Flatten(input_shape=[28, 28]),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
   
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])    

model.summary()  # 打印模型
keras.utils.plot_model(model, "model_info.png", show_shapes=True)  # 暂时还不能绘制模型结构图，需要安装pydot/graphviz

# %% 训练
"""训练指令model.fit(imgs, labels, epochs, batch_size, validation_data) 
1. 输入样本格式: (b, c, h, w) 或 (b, h, w)
2. 输入标签格式: (b,)
3. 
"""
# (60000, 28, 28) (60000,)作为输入，不考虑交叉熵要求的标签独热编码化吗？
model.fit(train_images, train_labels, epochs=5)  

# %% 验证
model.evaluate(test_images, test_labels)   # 传入测试数据，获得在测试数据集上的损失和精度

# %% 测试
predictions = model.predict(test_images)   # 一次预测多个样本
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

