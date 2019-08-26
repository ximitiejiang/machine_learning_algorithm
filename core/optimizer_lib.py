#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:16:54 2019

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# %% 正则化
class no_regularization():
    """无正则化"""
    def __init__(self, weight_decay=0.1):
        pass
    
    def __call__(self, w):
        return 0
    
    def gradient(self, w):
        return 0        


class l1_regularization():
    """l1正则化"""
    def __init__(self, weight_decay=0.1):
        self.weight_decay = weight_decay
        
    def __call__(self, w):
        return self.weight_decay * np.linalg.norm(w, 1)
    
    def gradient(self, w):
        return self.weight_decay * np.sign(w)


class l2_regularization():
    """l2正则化"""
    def __init__(self, weight_decay=0.1):
        self.weight_decay = weight_decay
        
    def __call__(self, w):
        return self.weight_decay * 0.5 * np.dot(w.T, w)
    
    def gradient(self, w):
        return self.weight_decay * w   # 对原函数直接对w求导    
    

# %% 优化器(用于更新权重): 优化器的核心是求函数loss(w)的极值，通过动态调整w的值(沿着梯度方向每次调整一点点)来求得loss的最小值。
class Optimizer():
    """优化器基类："""
    regularization_dict = {'l1': l1_regularization,
                           'l2': l2_regularization,
                           'no': no_regularization}
    
    def __init__(self, weight_decay=0.1, regularization_type='l2'):
        self.regularization_type = regularization_type 
        self.weight_decay = weight_decay
        
        if self.regularization_type is None or self.regularization_type == 'no':
            self.regularization = self.regularization_dict['no']()
        else:
            self.regularization = self.regularization_dict[regularization_type]()
        
    def update(self):
        raise NotImplementedError()


class SGD(Optimizer):
    """普通SGD梯度下降算法"""
    def __init__(self, lr=0.001, 
                 weight_decay=0.1, regularization_type='l2'):
        super().__init__(weight_decay=weight_decay, regularization_type=regularization_type)
        self.lr = lr
    
    def update(self, w, grad):
        grad += self.regularization(w)
        w -= self.lr * grad
        return w


class SGDM():
    """SGDM梯度下降算法-带动量M的SGD算法"""
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.tmp_w = None
    
    def update(self, w, grad):
        if self.tmp_w is None:
            self.tmp_w = np.zeros_like(w)
        self.tmp_w = self.momentum * self.tmp_w + (1 - self.momentum) * grad  # 计算更新m
        w -= self.lr * self.tmp_w
        return w


class AdaGrad():
    """自适应梯度调节："""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.G = None # Sum of squares of the gradients
        self.eps = 1e-8

    def update(self, w, grad):
        # If not initialized
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad, 2)
        # Adaptive gradient with higher learning rate for sparse data
        return w - self.lr * grad / np.sqrt(self.G + self.eps)


class RMSprop():
    """"""
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = None # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_wrt_w))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate *  grad_wrt_w / np.sqrt(self.Eg + self.eps)


class Adam():
    """Adam优化器：对梯度的一阶矩估计(梯度的均值)和二阶距估计(梯度的未中心化方差)进行综合考虑来更新步长。
    他是基于AdaGrad和RMSprop进行优化的产物。
    Args:
        lr：学习率
        b1/b2: 矩估计的指数衰减率 
    """
    def __init__(self, lr=0.001, b1=0.9, b2=0.999):
        self.lr = lr
        self.eps = 1e-8
        self.m = None  # 梯度的一阶矩
        self.v = None  # 梯度的二阶距
        self.b1 = b1
        self.b2 = b2
    
    def update(self, w, grad):
        if self.m is None:
            self.m = np.zeros(w.shape)  # 初始化一阶矩为0
            self.v = np.zeros(w.shape)  # 初始化二阶距为0
        self.m = self.b1 * self.m + (1 - self.b1) * grad               # 一阶矩迭代更新mt = b1*mt-1 + (1-b1)*g
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad, 2)  # 二阶距迭代更新vt = b2*bt-1 + (1-b2)*g^2
        
        m_hat = self.m / (1 - self.b1)    # 计算偏置修正一阶矩mt' = mt/(1-b1)
        v_hat = self.v / (1 - self.b2)    # 计算偏置修正二阶距vt' = vt/(1-b2)
        self.w_update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # 更新参数w = w - lr*mt'/(sqrt(vt') + eps)
        w -= self.w_update
        return w


class NM():
    """NM=NewtonMethod牛顿法: 参数更新公式变为x_k+1 = x_k - H_xk(-1) * grad, 其中H_xk(-1)代表海森矩阵的逆矩阵
    参考：https://blog.csdn.net/golden1314521/article/details/46225289，https://www.cnblogs.com/shixiangwan/p/7532830.html
    """
    def __init__(self):
        pass
    def update(self):
        pass
    
    
class BFGS():
    """BFGS拟牛顿法: """
    def __init__(self):
        pass
    def update(self):
        pass
    
    


    
# %% 对优化器进行测试
def test_optimizer():
    """采用二元函数测试优化器算法"""
    def f(x, y):    # 目标：求解一个函数的极小值，等价于在参数x,y变化情况下得到min f
        return x**2 / 20.0 + y**2

    def df(x, y):
        return x / 10.0, 2.0*y

    optimizers = OrderedDict()
    optimizers["SGD"] = SGD(lr=0.95)
    optimizers["SGDM"] = SGDM(lr=0.95, momentum=0.9)
    optimizers["AdaGrad"] = AdaGrad(lr=1.5)
    optimizers["Adam"] = Adam(lr=0.3)
    
    idx = 1
    for key in optimizers:
        optimizer = optimizers[key]
        x_hist = []
        y_hist = []
        params = np.array([-7.0, 2.0])  # 指定一个初始x,y参数位置 
        grads = np.ones((2,))       # 指定初始梯度
        for i in range(30):
            x_hist.append(params[0])
            y_hist.append(params[1])
            grads[0], grads[1] = df(params[0], params[1])
            params = optimizer.update(params, grads)
        
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)
        X, Y = np.meshgrid(x, y) 
        Z = f(X, Y)
        
        # for simple contour line  
        mask = Z > 7
        Z[mask] = 0

        plt.subplot(2, 2, idx)
        idx += 1
        plt.plot(x_hist, y_hist, 'o-', color="red") # 绘制所有x,y点
        plt.contour(X, Y, Z)  # 绘制原函数的等高线
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        plt.title(key)
#        plt.xlabel("x")
        plt.ylabel("y")
        
    plt.show()
    

def get_min_value():
    """求解一个一元函数的极值"""
    def fn(x):
        return (x-1)**2 - 1
    def df(x):
        return 2*(x-1)
    optimizer = SGD(lr=0.1)
    params = -10
    grads = 1
    x_hist = []
    n_epochs = 100
    for _ in range(n_epochs):
        grads = df(params)
        params = optimizer.update(params, grads)
        x_hist.append(params)
    min_value = fn(x_hist[-1])
    
    print("min fn: %f, x value: %f"%(min_value, x_hist[-1]))
    
    

if __name__ == "__main__":
    """尝试直接用optimizer做无约束极值问题"""
    id = 'all'
    
    if id == 'all':
        test_optimizer()
        
    if id == 'min':
        get_min_value()
