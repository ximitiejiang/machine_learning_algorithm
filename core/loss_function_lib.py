#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:01:00 2019

@author: ubuntu
"""

import numpy as np

# %% 损失函数
class Loss():
    """创建一个损失基类，仅用来汇总共有参数"""
    def loss(self, y, p):
        raise NotImplementedError()
    
    def gradient(self, y, p):
        raise NotImplementedError()
    
    def acc(self, y, p):  # 该acc函数可用于每个epoch输出一个acc，用于评价模型是否过拟合或者欠拟合：train_acc vs test_acc，从而在训练过程中就能评估出来。
        p = np.argmax(p, axis=1)  # (1280,)
        y = np.argmax(y, axis=1)  # (1280,)
        acc = np.sum(p == y, axis=0) / len(y)
        return acc


# %% 分类损失
class CrossEntropy(Loss):
    """交叉熵损失函数，通过评价两个分布y,y'的相关性来评价分类结果的准确性，而相关性指标采用交叉熵
    注意：这是原始交叉熵公式，所以对于y,p都必须是概率输入，即y是独热编码形式的概率，p是softmax或sigmoid输出形式的概率。
    
    公式：loss = -(y*log(y') + (1-y)*log(1-y'))
    梯度：loss对y'求导 = -y/y' - (1-y)/(1-y')
    """
    def __init__(self): 
        pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)  # 计算损失loss = -(t*log(p) + (1-t)*log(1-p))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)  # 计算该级梯度grad = loss' = -y/p + (1-y)/(1-p)


class FocalLoss(Loss):
    """focal loss多用于物体检测的分类部分: 本质是在交叉熵损失函数基础上进行优化。
    由于在检测任务中存在样本不平衡问题，即负样本远远多于正样本，从而各个权重的梯度方向基本被负样本主导，
    而得不到正确的梯度更新方向，这是因为w = w -lr*grad, 而这个grad在SGDM中是历史梯度的累积之和，如果都是
    负样本生成的其他方向梯度，累积之后的grad也会很大导致即使来了一个正样本产生的梯度也不足以把方向拉回到
    正确的梯度下降方向，所以导致loss无法收敛到最小值。
    公式: loss = 
    
    """
    def __init__(self, gamma):
        pass
    
    def loss(self, y, p):
        pass
    
    def gradient(self, y, p):
        pass



def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='elementwise_mean'):
    """带sigmoid的focal loss实现：
    """
    pred_sigmoid = pred.sigmoid()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)  # pt = (1-p)*
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    return F.binary_cross_entropy_with_logits(
        pred, target, weight, reduction=reduction)


# %% 回归损失
class L2Loss(Loss):
    """平方损失函数(即l2 loss)，通过评价两个分布y,y'的空间距离来评价回归结果的准确性。
    平方损失类似于所谓的mse()损失函数，只不过mse()是对平方损失进行缩减得到一个单损失值，
    而平方损失跟交叉熵损失都是直接输出包含每个元素的损失矩阵。
    平方损失优点是全区间都可导，且收敛更快，因为他对误差的惩罚更大，
    缺点是在训练初期预测偏差较大时更有可能造成训练不稳定或者梯度爆炸。
    公式：loss = 0.5*(y-y')^2
    梯度：loss对y'求导 = -(y-y')
    """
    def __init__(self): 
        pass
    
    def loss(self, y, p):
        loss_result = 0.5 * np.power((y - p), 2)  # 增加一个0.5是为了让梯度结果更简洁，直接抵消平方求导产生的2
        return loss_result
    
    def gradient(self, y, p):
        grad_result = -(y - p)
        return grad_result
    
    def acc(self, y, p): # 平方损失用来做回归，所以没有计算acc的必要，直接返回0
        return 0


class L1Loss(Loss):
    """常规l1 loss，也就是绝对值损失
    普通绝对值损失的优点是惩罚相对柔和，在初期训练稳定不容易梯度爆炸。
    缺点是在训练后期梯度不能进一步减小，无法达到更高精度。且在0点处不可导，从而导致该点梯度会产生一个突变(从-1变到+1)造成训练不稳定
    公式：loss = |y-y'|
    梯度：loss对y'求导 = 1(当delta>0), -1(当delta<0)
    """
    def __init__(self):
        pass
    
    def loss(self, y, p):
        diff = np.abs(p - y)
        return diff
    
    def gradient(self, y, p):
        diff = np.abs(p - y)
        return np.where(diff < 0, -1, 1)

class SmoothL1Loss(Loss):
    """带平滑区域的l1 loss，参考：https://www.zhihu.com/question/58200555
    类似于集成l1和l2的一个loss, 在中间段是l2, 在两头是l1.
    平滑l1损失的优点是在初期即使变化较大，其损失也固定为1或-1不会过大造成梯度爆炸，而在
    后期预测误差很小也能够让梯度进一步减小，可达到更高精度。
    其中l1,l2的转换点叫beta,当前主流是取beta=1/9或者简单取1也可以。
    公式：loss = |y-y'|^2/(2*beta) (当-beta<delta<beta), |y-y'|-0.5*beta (当delta>=beta, delta<=-beta)
    梯度：loss对y'求导 = 
    """
    def __init__(self, beta=1/9):  # beta值主流做法(比如pytorch采用1/9)
        self.beta = beta
        
    def loss(self, y, p):
        diff = np.abs(p - y)
        loss_result = np.where(diff < self.beta, 0.5 * diff * diff / self.beta, 
                               diff - 0.5 * self.beta)
        return loss_result
    
    def gradient(self, y, p):
        diff = np.abs(p - y)
        return np.where(diff < self.beta, - diff / self.beta, np.sign(p - y))
        
    
    
    
    