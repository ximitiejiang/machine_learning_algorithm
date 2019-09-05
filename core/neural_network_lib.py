#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:22:32 2019

mainly refer to https://github.com/eriklindernoren/ML-From-Scratch

@author: ubuntu
"""

import numpy as np
import math
import copy
import matplotlib.pyplot as plt


from core.base_model import BaseModel
from core.activation_function_lib import Relu, LeakyRelu, Elu, Sigmoid, Softmax, Tanh
from core.loss_function_lib import CrossEntropy, L2Loss, L1Loss, SmoothL1Loss
from core.optimizer_lib import SGD, Adam, SGDM, AdaGrad, RMSprop
from utils.dataloader import batch_iterator, train_test_split
from utils.matrix_operation import img2col, col2img
from dataset.digits_dataset import DigitsDataset
from dataset.mnist_dataset import MnistDataset
from dataset.regression_dataset import RegressionDataset

# %%  模型总成
class NeuralNetwork(BaseModel):
    """神经网络模型：这个NN是一个空壳类，并不包含任何层，用于模型创建的基类。
    Args:
        feats: (n_sample, n_feats) 
        labels: (n_sample, n_classes) 独热编码形式
        loss: 传入loss对象，该对象实现了__call__方法可直接调用
        optimizer: 传入optimizer对象，该对象主要是采用update(w, grad)方法
        model_type: sl(shallow learning/supervised learning) or dl(deep learning)
    """
    def __init__(self, feats, labels, 
                 loss, optimizer, n_epochs, batch_size, 
                 val_feats=None, val_labels=None):
        super().__init__(feats, labels)
        
        self.optimizer = optimizer
        self.loss_function = loss
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.layers = []
        self.val_feats = val_feats   # 用于每个batch的acc验证
        self.val_labels = val_labels
        self.trained = False  # 初始状态为未训练的无参状态，不能进行预测和验证

    def add(self, layer):
        """用于模型添加层: 设置输入输出层数，初始化"""      
        if hasattr(layer, "initialize"):
            layer.initialize(optimizer = self.optimizer)
        self.layers.append(layer)
        # 同时设置一下输入的形状：否则batchnorm层拿不到前一层的输入shape
        
    
    def forward_pass(self, x, training):
        """前向计算每一层的输出"""
        for layer in self.layers:
            layer_output = layer.forward_pass(x, training=training)
            x = layer_output
        return layer_output
    
    def backward_pass(self, grad):
        """基于梯度反向更新每一层的累积梯度同时更新每一层的梯度"""
        for layer in self.layers[::-1]:
            grad = layer.backward_pass(grad)  # 梯度的反向传播，且传播的是每一层的累积梯度
            
    def batch_operation_train(self, x, y):
        """基于每一个batch的数据分别进行前向计算和反向计算"""
        # 前向计算
        y_pred = self.forward_pass(x=x, training=True)
        losses = self.loss_function.loss(y, y_pred)
        loss = np.mean(losses)
        acc = self.loss_function.acc(y, y_pred)
        # 反向传播
        loss_grad = self.loss_function.gradient(y, y_pred)
        self.backward_pass(grad = loss_grad)
        return loss, acc
    
    def batch_operation_val(self, val_x, val_y):
        """在每个batch做完都做一次验证，比较费时，但可实时看到验证集的acc变化，可用于评估是否过拟合。
        注意：val操作时跟test一样，不需要反向传播。
        """
        # BN和dropout在训练和验证过程中需要有不同的操作差异，(其他层不受影响)，所以在val和test时，需要闯入training flag给每一个层。      
        y_pred = self.forward_pass(val_x, training=False) # 在验证时是假定已经训练完成，但由于该步是在train中执行，必须手动设置trained=False
        losses = self.loss_function.loss(val_y, y_pred)
        loss = np.mean(losses)
        acc = self.loss_function.acc(val_y, y_pred)
        return loss, acc
        
    def train(self):
        total_iter = 1
        all_losses = {'train':[], 'val':[]}
        all_accs = {'train':[], 'val':[]}
        for i in range(self.n_epochs):
            it = 1
            for x_batch, y_batch in batch_iterator(self.feats, self.labels, 
                                                   batch_size = self.batch_size):
                # 训练数据的计算
                loss, acc = self.batch_operation_train(x_batch, y_batch)
                all_losses['train'].append([total_iter,loss])
                all_accs['train'].append([total_iter, acc])
                log_text = "iter %d/epoch %d: batch_loss=%f, batch_acc=%f"%(it, i+1, loss, acc)
                # 验证数据的计算
                if self.val_feats is not None:
                    val_loss, val_acc = self.batch_operation_val(self.val_feats, self.val_labels)
                    all_losses['val'].append([total_iter, val_loss])
                    all_accs['val'].append([total_iter, val_acc])
                    log_text += ", val_loss=%f, val_acc=%f"%(val_loss, val_acc)
                # 显示loss
                if it % 5 == 0:
                    print(log_text)

                it += 1
                total_iter += 1
                
        self.vis_loss(all_losses['train'], all_accs['train'], title='train')
        if self.val_feats is not None:
            self.vis_loss(all_losses['val'], all_accs['val'], title='val')
        self.trained = True  # 完成training则可以预测
        return all_losses, all_accs
    
    def summary(self):
        """用来显示模型结构"""
        pass
    
    def evaluation(self, x, y, title=None):
        """对一组数据进行预测精度"""
        if title is None:
            title = "evaluate"
        if self.trained:
            y_pred = self.forward_pass(x, training=False)  # (1280, 10)
            loss = self.loss_function.loss(y, y_pred).mean()
            acc = self.loss_function.acc(y, y_pred)
            
            # 比较一下如果用train模块出来的acc
            y_pred1 = self.forward_pass(x=x, training=True)
            losses1 = self.loss_function.loss(y, y_pred1)
            loss1 = np.mean(losses1)
            acc1 = self.loss_function.acc(y, y_pred1)
            print('training params acc: %f, loss: %f'%(acc1, loss1))
            
            print(title + " acc: %f, loss: %f"%(acc, loss))
            return acc, y_pred
        else:
            raise ValueError("model not trained, can not predict or evaluate.")
        
    
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
        
        self.add(Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=1))# w'=(8-3+2)/1 +1=8
        self.add(Activation('relu'))
        self.add(BatchNorm2d(16))
        self.add(Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=1))# w'=(8-3+2)/1 +1=8
        self.add(Activation('relu'))
        self.add(BatchNorm2d(32))
        self.add(Flatten())  # (64,32,8,8) -> (64,32*8*8=2048)
        self.add(Linear(in_features=2048, out_features=256))  #
        self.add(Activation('relu'))
        self.add(BatchNorm2d(256))
        self.add(Linear(in_features=256, out_features=10))
        self.add(Activation('softmax'))


class CNN3(NeuralNetwork):
    """一个完整包含3种典型层(卷积+池化+全连接)的卷积神经网络
    在cpu上跑包含卷积的模型，由于卷积需要的计算量和内存空间都远超过全连接，所以不能训练大图。
    只能在digits这种8*8的小图上跑比较合适，连mnist这种28*28的图都会很慢。
    """
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size,
                 val_feats=None, val_labels=None):
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size,
                         val_feats=val_feats,
                         val_labels=val_labels)
        # 假定输入(b,1,8,8), 卷积后w = (8-3+2)/1 +1=8         
        self.add(Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=1)) #(b,16,8,8)
        self.add(Activation('relu'))
        self.add(BatchNorm2d(16))
        # 池化后w= (28-2+0)/2 + 1=14
        self.add(MaxPool2d(kernel_size=(2,2), stride=2, padding=0))# (b,16,4,4)
        self.add(Flatten())   # (b,256)
        self.add(Linear(in_features=256, out_features=256)) # (b,256)
        self.add(Activation('relu'))
        self.add(BatchNorm2d(256)) 
        self.add(Linear(in_features=256, out_features=10))  # (b, 10)
        self.add(Activation('softmax'))
        
    
class MLP(NeuralNetwork):
    """基于神经网络结构的多层(2层)感知机"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(feats=feats, labels=labels, 
                         loss=loss,
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
        n_feats = feats.shape[1]
        n_classes = labels.shape[1]
        self.add(Linear(in_features=n_feats, out_features=16))
        self.add(Activation('sigmoid'))
        self.add(Linear(in_features=16, out_features=n_classes))
        self.add(Activation('softmax'))


class SoftmaxReg(NeuralNetwork):
    """基于神经网络结构的多分类softmax模型"""
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(feats=feats, labels=labels, 
                         loss=loss, 
                         optimizer=optimizer, 
                         n_epochs=n_epochs, 
                         batch_size=batch_size)
        n_feats = feats.shape[1]
        n_classes = labels.shape[1]
        self.add(Linear(in_features=n_feats, out_features=n_classes))
        self.add(Activation('softmax'))


class LinearRegression(NeuralNetwork):
    """回归模型: 本质上跟分类模型一样，都是通过损失函数来评价预测值y'与真值y之间的相关性。
    线性回归模型本质上其实就是一层线性层，直接用输出作为回归预测，不需要任何激活函数。
    """
    def __init__(self, feats, labels, loss, optimizer, n_epochs, batch_size):
        
        super().__init__(feats=feats, labels=labels, 
                       loss=loss, 
                       optimizer=optimizer, 
                       n_epochs=n_epochs, 
                       batch_size=batch_size)
        self.add(Linear(in_features=1, out_features=1))
    
    def evaluation(self, x, y, title=None):
        """回归的评估：没有acc可评估，直接绘制拟合曲线"""
        y_pred = self.forward_pass(x, training=False)
        plt.figure()
        if title is None:
            title = 'regression curve'
        plt.title(title)
        plt.scatter(x, y, label='raw data', color='red')
        plt.plot(x, y_pred, label = 'y_pred', color='green')
        plt.legend()
        plt.grid()        


# %% 单层模型
class Layer():
    """层的基类: 该基类不需要初始化，用于强制定义需要实施的方法，并把所有共有函数放在一起"""
    def forward_pass(self, x, training):
        raise NotImplementedError()
    
    def backward_pass(self, accum_grad):
        raise NotImplementedError()
    
    def input_shape(self):
        pass
    
    def output_shape(self):
        pass
    
    
class Linear(Layer):
    """全连接层: 要求输入必须是经过flatten的二维数据(n_samples, n_feats)"""
    def __init__(self, in_features, out_features):  
        self.out_features = out_features  # (m,)
        self.in_features = in_features    # (n,)
        self.trainable = True
        self.W = None
        self.W0 = None
    
    def initialize(self, optimizer):
        limit = math.sqrt(2.) / math.sqrt(self.in_features)  # 从xavier初始化改成了kaiming初始化
        self.W = np.random.uniform(-limit, limit, (self.in_features, self.out_features))
        self.W0 = np.zeros((1, self.out_features))  # TODO  w0(1,10)
        
        self.W_optimizer = copy.copy(optimizer)
        self.W0_optimizer = copy.copy(optimizer)
    
    def forward_pass(self, x, training=True):
        self.input_feats = x     # 保存输入，为反传预留
        return np.dot(x, self.W) + self.W0  # (8,64)(64,10)+(1,10) ->(8,10)+(1,10)
    
    def backward_pass(self, accum_grad): 
        tmp_W = self.W
        # 更新参数
        if self.trainable:
            grad_w = np.dot(self.input_feats.T, accum_grad)  # 关键1:计算权值梯度时是对w求导得到的是x所以是累积梯度乘以输入x，(8,64).T (8,10) -> (64,10)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)   # (64,10)->(1,10)  TODO: 偏置参数的更新逻辑
            
            self.W = self.W_optimizer.update(self.W, grad_w)      
            self.W0 = self.W0_optimizer.update(self.W0, grad_w0)
        # 累积梯度
        accum_grad = np.dot(accum_grad, tmp_W.T)           # 关键2:计算累积梯度时是对x求导得到的是w所以是累积梯度乘以权值w  
        return accum_grad
    
        
class Conv2d(Layer):
    """卷积层
    注意：在img2col/col2img以及结果reshape时，要确保合并的时候维度是什么顺序，
    分离的时候就必须用相同的顺序，否则可能造成结果不匹配。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def initialize(self, optimizer):
        h, w = self.kernel_size
        
        limit = 1 / math.sqrt(np.prod(self.kernel_size))
        # 初始化卷积参数w和偏置w0: 卷积采用xavier初始化法，偏置初始化为0
        self.W = np.random.uniform(-limit, limit, size=(self.out_channels ,self.in_channels, h, w))  # (16,1,3,3)
        self.W0 = np.zeros((self.out_channels, 1)) #(16,1)
        self.W_optimizer = copy.copy(optimizer)
        self.W0_optimizer = copy.copy(optimizer)
    
    def forward_pass(self, x, training=True):
        self.input_feats = x  # 为反传预存x
        filter_h, filter_w = self.kernel_size
        out_h = (x.shape[2] + 2*self.padding - filter_h) // self.stride + 1  # 计算输出图像实际尺寸
        out_w = (x.shape[3] + 2*self.padding - filter_w) // self.stride + 1
        batch_size = x.shape[0]
        # 特征数据从matrix(b,c,h,w)变为列数据(卷积核元素个数，卷积次数), 每列就是一个卷积核的数据
        #img(b,c,h,w)变成col(c*kerh*kerw, b*oh*ow)，其中c*kerh*kerw代表一个立体卷积核元素个数，b*oh*ow代表卷积次数
        self.x_col = img2col(x, filter_h, filter_w, 
                             self.stride, self.padding) # (9，4096)表示每个滤波器要跟图像进行的滤波次数为4096，所以把每组9元素都准备出来。
        self.w_col = self.W.reshape(self.out_channels, -1)  # (16, 9)
        output = np.dot(self.w_col, self.x_col) + self.W0  #(16,9)*(9,4096)+(16,1) -> (16,4096) 列形式的w点积列形式的特征x
        output = output.reshape(self.out_channels,  out_h, out_w, batch_size)  # (16,8,8,64)
        # Redistribute axises so that batch size comes first
        return output.transpose(3,0,1,2)  #(64,16,8,8)
    
    def backward_pass(self, accum_grad):
        # 把累积梯度先变换成col的形式
        accum_grad = accum_grad.transpose(1,2,3,0).reshape(self.out_channels, -1) # (64,32,8,8)->(32,8,8,64)->(32, 4096)
        # 先获得反传梯度的分支到w
        grad_w = np.dot(accum_grad, self.x_col.T).reshape(self.W.shape)  # (32,4096)dot(4096,144)->(32,144)->(32,16,3,3)
        grad_w0 = np.sum(accum_grad, axis=1, keepdims=True) # ()
        # 更新w
        self.W = self.W_optimizer.update(self.W, grad_w)
        self.W0 = self.W0_optimizer.update(self.W0, grad_w0)
        # 继续反传
        accum_grad = np.dot(self.w_col.T, accum_grad)  # (32,144).T dot(32,4096)->(144,4096)
        # col形式转换回matrix
        filter_h, filter_w = self.kernel_size
        accum_grad = col2img(accum_grad, self.input_feats.shape,
                             filter_h, filter_w,
                             self.stride, self.padding)
        return accum_grad


class Activation(Layer):
    """激活层: 基于输入的激活函数类型name来生成激活函数对象进行调用"""
    activation_dict = {'relu': Relu,
                       'sigmoid': Sigmoid,
                       'softmax': Softmax,
                       'tanh': Tanh,
                       'leaky_relu': LeakyRelu,
                       'elu': Elu}
    def __init__(self, name):
        self.name = name
        self.activation_func = self.activation_dict[name]()  # 创建激活函数对象
        self.trainable = True
    
    def forward_pass(self, x, training=True):
        self.input_feats = x
        return self.activation_func(x)
    
    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.input_feats)
    
        
    
class BatchNorm2d(Layer):
    """归一化层
    注意：bn层在训练时需要实时更新mean和var
    但在测试时输入的特征不再用来更新mean,var，而是直接用训练时的mean,var做前向计算即可。
    """
    def __init__(self, in_features, momentum=0.99):  #参考pytorch的接口
        self.in_features = in_features  # 如果在卷积后边，这里in_features就是通道数C,如果在全连接后边，这里in_features就是特征列数
        self.momentum = momentum  #
        self.running_mean = None # 均值: 用来保存训练的均值，便于在测试环节使用
        self.running_var = None # 方差: 用来保存训练的方差，便于在测试环节使用
        self.eps = 0.01
    
    def initialize(self, optimizer):
        self.gamma = None   # 由于代码结构问题这里拿不到输入特征形状，无法初始化，改在在forward_pass初始化
        self.beta = None
        
        self.gamma_optimizer = copy.copy(optimizer)
        self.beta_optimizer = copy.copy(optimizer)
    
    def forward_pass(self, x, training):
        if self.gamma is None:
            self.gamma = np.ones((x.shape[1:]))  # 每个batch(c,h,w)乘以gamma, 所以在卷积后边(c,h,w) or 在全连接后边(n,)
            self.beta = np.zeros((x.shape[1:]))
        # 计算初始均值
        if self.running_mean is None:
            self.running_mean = np.mean(x, axis=0)  # 输入x(b,c,h,w)或(b,n)，均值为一张图整体均值，所以axis=0
            self.running_var = np.var(x, axis=0)
        if training: # 如果是训练状态
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean  # 更新时采用移动平均
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:   # 如果是val状态或者test状态，输入的数据不再拿来更新模型的mean和std，而是用模型训练数据的mean,std
            mean = self.running_mean
            var = self.running_var
        # 基于当前mean,var计算变换后的x = (x-mean)/std
        self.x_ctr = x - mean  # x-mean表示变换后的中心
        self.std_inv = 1 / np.sqrt(var + self.eps)  # 1/std 表示标准差分之一
        x_norm = self.x_ctr * self.std_inv        # 得到标准化输出x=(x-mean)*(1/std)   (64,16,8,8)
        output = self.gamma * x_norm + self.beta  # 得到gamma*x + beta
        return output
        
        
    def backward_pass(self, accum_grad):
        # 预存一个gamma
        tmp_gamma = self.gamma
        #
        x_norm = self.x_ctr * self.std_inv  # x = (x-mean)/std
        # 计算gamma和beta的梯度
        grad_gamma = np.sum(accum_grad * x_norm, axis=0)  #dloss/dgamma = (dloss/dy)*(dy/dgamma)=accum_grad * x_norm
        grad_beta = np.sum(accum_grad, axis=0)            #dloss/dbata = (dloss/dy)*(dy/dgamma)=accum_grad * 1
        # 基于梯度更新gamma, beta
        self.gamma = self.gamma_optimizer.update(self.gamma, grad_gamma)  
        self.beta = self.beta_optimizer.update(self.beta, grad_beta)
        
        batch_size = accum_grad.shape[0]
        # dloss/dx = (dloss/dx_norm)*(dx_norm/dx) 
        accum_grad = (1 / batch_size) * tmp_gamma * self.std_inv * \
            (batch_size * accum_grad - np.sum(accum_grad, axis=0)  \
            - self.x_ctr * self.std_inv**2 * np.sum(accum_grad * self.x_ctr, axis=0))
        
        return accum_grad
        

class Dropout(Layer):
    """关闭神经元层"""
    def __init__(self, p=0.3):  # p代表关闭的可能性
        self.p = p
        self.mask = None
    
    def forward_pass(self, x, training):
        if training:  # 训练过程中
            self.mask = np.random.uniform(size=x.shape) > self.p  # 0-1均匀分布，大于0.3为真，所以1表示打开，0表示关闭
            close = self.mask    # 训练时x*close则部分输出直接变为0
        else:   # val或test状态
            close = 1 - self.p   # val/test时则不再直接置0，而是只去数值的百分之(1-p)，比如关闭率0.3，则取70%做val/test
        return x * close
            
    def backward_pass(self, accum_grad):
        return accum_grad * self.mask

    
class Flatten(Layer):
    """展平层: 默认展平成(b, c*h*w)
    """
    def __init__(self):
        self.prev_shape = None
    
    def forward_pass(self, x, training=True):
        self.prev_shape = x.shape  # 预留输入的维度信息为反传做准备
        return x.reshape(x.shape[0], -1)
    
    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape) 


class MaxPool2d(Layer):
    """最大池化层: 通过池化核提取区域最大值，相当于一方面提取区域特征，另一方面可缩小特征图尺寸，扩大感受野
    注意：最大池化不能更改通道数，只能更改输出特征图尺寸oh,ow, 所以只能用来进行感受野扩展
    """
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward_pass(self, x, training=True):
        self.batch_size, self.channels, self.in_h, self.in_w = x.shape
        f_h, f_w = self.kernel_size
        
        out_h = (self.in_h + 2*self.padding - f_h) // self.stride + 1  # 计算输出图像实际尺寸
        out_w = (self.in_w + 2*self.padding - f_w) // self.stride + 1
        # 由于maxpool的核是二维核，只针对一个通道的平面数据进行取最大值(相比之下卷积核是三维核可针对输入多通道一次完成卷积)
        # 所以maxpool列表化时需要把b和c合在一起，把层数定义成1
        x = x.reshape(self.batch_size*self.channels, 1, self.in_h, self.in_w)
        x_col = img2col(x, f_h, f_w, self.stride, self.padding)  # 变成col(c*kerh*kerw, b*oh*ow)=(4,64*16*14*14=200704) 
        # 每一列提取最大值idx(0到8),并作为self变量预存
        self.arg_max = np.argmax(x_col, axis=0).flatten()  # (16384,)
        # 提取最大值
        output = x_col[self.arg_max, range(self.arg_max.size)] # 两个维度同时索引(注意不是切片)，得到交叉点，等效于取每一列的最大值。
        # col转换成matrix (200704)
        output = output.reshape(out_h, out_w, self.batch_size, self.channels).transpose(2, 3, 0, 1)  # (b,c,oh,ow)
        return output
        
    def backward_pass(self, accum_grad):
        # maxpool的操作是提取最大值，这种方法是不能直接求导的，也就不能直接通过导数公式反传梯度
        # 解决方式是把每个梯度值放入原grad_w的对应位置，其他没有对应梯度的位置全都置0，保证前后梯度的总和是一样的。
        accum_grad = accum_grad.transpose(2, 3, 0, 1).flatten()  # (b,c,oh,ow) -> (oh,ow,b,c)->(oh*ow*b*c)
        # 创建全0列元素, 把max_id对应位置写入grad_w梯度，其他位置保持0
        accum_grad_col = np.zeros((np.prod(self.kernel_size), accum_grad.size)) # 每一个梯度都是由一个kernel的所有元素生成，所以先得到一组全0
        accum_grad_col[self.arg_max, range(accum_grad.size)] = accum_grad
        # 准备继续反传的累积梯度
        accum_grad = col2img(accum_grad_col, 
                             (self.batch_size * self.channels, 1, self.in_h, self.in_w), 
                             self.kernel_size[0], self.kernel_size[1],
                             self.stride, self.padding)
        accum_grad = accum_grad.reshape(self.batch_size, self.channels, self.in_h, self.in_w)
        return accum_grad


class AvgPool2d(Layer):
    """平均池化层: 通过池化核提取区域平均值，相当于一方面提取区域特征，另一方面可缩小特征图尺寸，扩大感受野
    注意：平均池化也不能更改通道数，只能更改输出特征图尺寸oh,ow, 所以只能用来进行感受野扩展
    """
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward_pass(self, x, training=True):
        self.batch_size, self.channels, self.in_h, self.in_w = x.shape
        f_h, f_w = self.kernel_size
        
        out_h = (self.in_h + 2*self.padding - f_h) // self.stride + 1  # 计算输出图像实际尺寸
        out_w = (self.in_w + 2*self.padding - f_w) // self.stride + 1
        x = x.reshape(self.batch_size*self.channels, 1, self.in_h, self.in_w)
        x_col = img2col(x, f_h, f_w, self.stride, self.padding)  # 变成col(c*kerh*kerw, b*oh*ow)
        # 这里跟maxpool不同: 对每列求平均值即可
        output = np.mean(x_col, axis=0)
        # col转换成matrix
        output = output.reshape(out_h, out_w, self.batch_size, self.channels).transpose(2, 3, 0, 1)  # (b,c,oh,ow)
        return output
        
    def backward_pass(self, accum_grad):
        # avg pool的操作是提取平均值，这种方法是不能直接求导的，也就不能直接通过导数公式反传梯度
        # 解决方式是把每个梯度值先平均一下，把平均值放入原grad_w的对应区域的所有位置，保证前后梯度的总和是一样的。
        accum_grad = accum_grad.reshape(2, 3, 0, 1).flatten()  # (b,c,oh,ow) -> (oh,ow,b,c)->(oh*ow*b*c)
        # 这里跟maxpool不同：需要把梯度先平均一下，然后用平均值写入grad_w的所有位置。
        accum_grad_col = np.zeros((np.prod(self.kernel_size), accum_grad.size)) # 每一个梯度都是由一个kernel的所有元素生成，所以先得到一组全0
        accum_grad_col[:, range(accum_grad.size)] = 1. / accum_grad_col.shape[0] * accum_grad
        # 准备继续反传的累积梯度
        accum_grad = col2img(accum_grad_col, 
                             (self.batch_size * self.channels, 1, self.in_h, self.in_w), 
                             self.kernel_size[0], self.kernel_size[1],
                             self.stride, self.padding)
        accum_grad = accum_grad.reshape(self.batch_size, self.channels, self.in_h, self.in_w)
        return accum_grad


class Empty(Layer):
    """空层: 用来占位置号"""
    def __init__(self):
        pass
    
    def forward_pass(self, x, training=True):
        return x
    
    def backward_pass(self, accum_grad):
        return accum_grad
    

# %% 调试
if __name__ == "__main__":
    
    model = 'cnn'
    
    if model == 'softmax':  # 输入图片是(b,n)
    
        dataset = DigitsDataset(norm=True, one_hot=True)
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels, test_size=0.3, shuffle=True)
        
        optimizer = SGDM(lr=0.001)
        loss_func = CrossEntropy()
        clf = SoftmaxReg(train_x, train_y, 
                         loss=loss_func, 
                         optimizer= optimizer, 
                         batch_size = 64, 
                         n_epochs=200)
        clf.train()
        acc1, _ = clf.evaluation(train_x, train_y)
        print("training acc: %f"%acc1)
        acc2, _ = clf.evaluation(test_x, test_y)
        print("test acc: %f"%acc2)
           
    if model == "mlp":  # 输入图片是(b,n)
        dataset = DigitsDataset(norm=True, one_hot=True)
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels, test_size=0.3, shuffle=True)
        
        optimizer = SGDM(lr=0.001, momentum=0.9)
#        optimizer = RMSprop(lr=0.001)
        loss_func = CrossEntropy()
        clf = MLP(train_x, train_y, 
                  loss=loss_func, 
                  optimizer= optimizer, 
                  batch_size = 64, 
                  n_epochs=200)
        clf.train()
        acc1, _ = clf.evaluation(train_x[:64], train_y, "train")
        acc2, _ = clf.evaluation(test_x[:64], test_y, "test")

    
    if model == "cnn":  # 输入图片必须是(b,c,h,w)
        dataset = DigitsDataset(norm=True, one_hot=True)
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels, test_size=0.1, shuffle=True)
#        test_x, val_x, test_y, val_y = train_test_split(test_x, test_y, test_size=0.3, shuffle=True)
        
        # 已展平数据转matrix
        train_x = train_x.reshape(-1, 1, 8, 8)
        test_x = test_x.reshape(-1, 1, 8, 8)
#        val_x = val_x.reshape(-1, 1, 8, 8)
        
        optimizer = SGDM(lr=0.01, momentum=0.9)
        loss_func = CrossEntropy()
        clf = CNN(train_x, train_y, 
                  loss=loss_func, 
                  optimizer= optimizer, 
                  batch_size = 64, 
                  n_epochs=5)
        clf.train()
#        acc1, _ = clf.evaluation(train_x, train_y, "train")

        acc, _ = clf.evaluation(test_x, test_y, "test")


    if model == 'cnn3':
        dataset = DigitsDataset(norm=True, one_hot=True)
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.datas, dataset.labels, test_size=0.3, shuffle=True)
#        test_x, val_x, test_y, val_y = train_test_split(test_x, test_y, test_size=0.3, shuffle=True)
        
        # 已展平数据转matrix
        train_x = train_x.reshape(-1, 1, 8, 8)
        test_x = test_x.reshape(-1, 1, 8, 8)
#        val_x = val_x.reshape(-1, 1, 28, 28)
        
        optimizer = SGDM(lr=0.01, momentum=0.9)
        loss_func = CrossEntropy()
        clf = CNN3(train_x, train_y, 
                  loss=loss_func, 
                  optimizer= optimizer, 
                  batch_size = 64, 
                  n_epochs=20)
        clf.train()
#        clf.evaluation(train_x, train_y, "train")
        clf.evaluation(test_x, test_y, "test") 

    
    if model == 'reg':
        dataset = RegressionDataset(n_samples=500, n_features=1, n_targets=1, noise=4)
        X = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)    
        
        train_x = train_x.reshape(-1, 1)
        test_x = test_x.reshape(-1, 1)
        train_y = train_y.reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)
        
        optimizer = SGDM(lr=0.0001, weight_decay=0.1, regularization_type='l2')  # TODO: 这里暂时只有SGD/SGDM能够收敛，Adam不能
#        loss_func = L2Loss()
        loss_func = SmoothL1Loss()  # 这里用smoothl1是收敛的，但l2loss暂时没法收敛
        reg = LinearRegression(train_x, train_y, 
                               loss=loss_func, 
                               optimizer= optimizer, 
                               batch_size = 64, 
                               n_epochs=500)
        reg.train()
        reg.evaluation(train_x, train_y, "train")
        reg.evaluation(test_x, test_y, "test")        
        
        
        
        