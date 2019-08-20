#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:22:32 2019

@author: ubuntu
"""
from core.base_model import BaseModel

class NeuralNetwork(BaseModel):
    
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        self.feats = feats
        self.labels = labels
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.layers = []

    def add(self, layer):
        """用于模型添加层: 设置输入输出层数，初始化"""
        if self.layers:
            pass
    
    def forward_pass(self, x):
        """前向计算每一层的输出"""
        layer_output = x
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output)
        return layer_output
    
    def backward_pass(self, grad):
        """基于梯度反向更新每一层的参数"""
        for layer in self.layers[::-1]:
            grad = layer.backward_pass(grad)
            
    def batch_op(self, x, y):
        """基于每一个batch的数据分别进行前向计算和反向计算"""
        y_pred = self.forward_pass(x)
        loss = self.loss_function.loss(y, y_pred)
        
        grad = self.loss_function.gradient(y, y_pred)
        self.backward_pass(grad = grad)
        
        return loss
    
    def train(self):
        assert self.layers, ""
        for _ in range(self.n_epochs):
            
            batch_error = []
            for x_batch, y_batch in get_batch_data(self.feats, self.labels, batch_size = self.batch_size):
                
                loss = self.batch_op()
                batch_error.append(loss)
        
        return self.errors
        
    
class FourLayersCNN(NeuralNetwork):
    """基于神经网络结构的多层卷积神经网络"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(self, feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
        
        self.add(Conv2D())
        
    
class MLP(NeuralNetwork):
    """基于神经网络结构的多层感知机"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(self, feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
    
    
##############################模型子层#########################################
class Layer():
    
    def calc_input_shape(self, input_shape):
        self
    
    def calc_output_shape(self):
        pass
    
class Conv2D(Layer):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def init_parameters(self):
        # 初始化卷积参数w
        self.w = np.random.uniform()
        # 初始化偏置参数b
        self.w0 = np.zeros()
    
    def forward_pass(self, x):
        pass
    
    def backward_pass(self, grad):
        pass
    
    def calc_output_shape(self):
        """实时计算层的输出数据结构: w' = (w - k_size + 2*pad) / stride +1"""
        output_height = 
        output_width = 
        return self.out_channels, output_height, output_width
    
    def calc_input_shape(self):
        """实时计算层的输入数据结构"""
    

###########################激活函数############################################

class Activation():
    
    def __init__(self, name):

        
class Sigmoid():
    """sigmoid函数的计算及梯度计算: 对sigmoid函数直接可求导"""
    def __call__(self, x):
        return 1 / (1 + np.exp(-x)) 
    
    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
    
class Softmax():
    """softmax函数的计算及梯度计算：对softmax函数直接可求导"""
    def __call__(self, x):
        exp_x = 
        return exp_x / np.sum(exp_x, axis=-1)
    
    def gradient(self, x):
        
        return 
    
    
class ReLU():
    """relu函数的计算及梯度计算：对relu函数直接可求导"""
    def __call__(self, x):
        return x if x>=0 else 0
    
    def gradient:
        
class TanH():
    """tanh函数的计算及梯度计算"""
class LeakyReLU():
    """leakyrelu函数的计算及梯度计算"""
class ELU():
    """elu函数的计算及梯度计算"""
    

#######################损失函数################################################
class CrossEntropy(Loss):
    def __init__(self): 
        pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
        

#######################梯度算法################################################
class Adam():
    """Adam优化器：对梯度的一阶矩估计(梯度的均值)和二阶距估计(梯度的未中心化方差)进行综合考虑来更新步长。
    Args:
        lr：学习率
        b1/b2: 矩估计的指数衰减率 
    """
    def __init__(self):
        pass


class SGD():
    """梯度下降算法-带动量的SGD"""
    def __init__(self):
        pass
    
    
#######################调试###################################################    
if __name__ == "__main__":
    
    train_x, test_x, train_y, test_y
    
    optimizer = Adam()
    loss_func = CrossEntropy()
    clf = FourLayersCNN(train_x, train_y, 
                        loss=loss_func, 
                        optimizer= optimizer, 
                        batch_size = 8, 
                        n_epochs=100)
    
    
    
    
    