#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 08:51:50 2018

@author: suliang
"""
import numpy as np

def AND(x1, x2):  # 与门
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    temp = sum(x * w) + b # 对应位置乘（只有mat才跟matlab一样是点积）
    if temp <=0:
        return 0
    else:
        return 1
    
def NAND(x1, x2):  # 与非门
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    temp = sum(x * w) + b # 对应位置乘（只有mat才跟matlab一样是点积）
    if temp <=0:
        return 0
    else:
        return 1
    
def OR(x1,x2):  # 或门
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    temp = sum(x * w) + b # 对应位置乘（只有mat才跟matlab一样是点积）
    if temp <=0:
        return 0
    else:
        return 1
    
def XOR(x1, x2):  # 异或门：不能用单层线性感知机实现
    x = np.array([x1, x2])
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    temp = AND(s1,s2)
    return temp


# ------def for NN--------------
def sigmoid(x):     # 逻辑函数
    return 1.0/(1+np.exp(-x))


def identity_function(x):   # 恒等函数
    return x


def relu(x):
    if x > 0: return x
    else: return 0 

def init_network():  # 定义网络参数：
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1      # 中间层激活函数都是逻辑函数
    z1 = sigmoid(a1)    
    
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)     # 输出层的激活函数是恒等函数

    return y


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   # 由于softmax函数的解的冗余性，减去最大值，防止溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))  # 为了防止y=0时出现负无限大

def loadDataSet(path, kind='train'):  # 导入数据，可分别指定导入train或者test数据集
    import os
    import struct
    import numpy as np
    # 拼接生成文件路径
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)    
    # 打开labels文件
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    # 打开images文件
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def drawNum(X,y):
    import matplotlib.pyplot as plt
    
    # 先绘制0-9
    fig, ax = plt.subplots(nrows=2, ncols=5,
                           sharex=True,sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    
    # 再绘制多个7看不同的写法有多少差别
    fig, ax = plt.subplots(nrows=4, ncols=5,
                           sharex=True,sharey=True, )
    ax = ax.flatten()
    for i in range(20):
        img = X[y == 6][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, 
                 weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
        
    def 





# ---main-------
path = '/Users/suliang/MyDatasets/MNIST'  # 绝对路径
images, labels = loadDataSet(path, kind='train')
drawNum(images, labels)


'''
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)


a = np.array([0.3,2.9,4.0])
y = softmax(a)
    
network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)
    

'''





   

    