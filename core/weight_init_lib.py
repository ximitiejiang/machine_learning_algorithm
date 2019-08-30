#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:38:49 2019

@author: ubuntu
"""
import numpy as np
#TODO: 是否采用uniform, 是否采用randn，均匀分布核正态分布的区别
def xavier_init(n, shape):
    """该初始化适合线性激活函数，比如sigmoid/tanh的中间部分接近线性函数。
    采用的是前一层节点个数n(或者卷积核h*w的乘积)，生成标准差sqrt(1/n)的分布。
    而对relu激活函数最好采用relu专用的kaiming_init
    参数：n为前一层节点数，shape为该层size
    """
    limit = 1 / np.sqrt(n)
    return np.random.uniform(-limit, limit, shape)
    

def kaiming_init(n, shape):
    """该初始化方法为专门配合激活函数relu.
    采用的是前一层节点个数n(或者卷积核h*w的乘积)，生成标准差sqrt(2/n)的分布。
    """
    limit = np.sqrt(2. / n)
    return np.random.uniform(-limit, limit, shape)

