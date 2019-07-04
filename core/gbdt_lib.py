#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:05:40 2019

@author: ubuntu
"""
from .base_model import BaseModel
from .decision_tree_lib import CART


class GBDT(CART):
    
    def __init__(self):
        """GBDT梯度提升树算法特点：属于boost算法的一部分，所以总思路跟ada boost有类似地方，是串行训练多个模型，
        每个模型会在前一个模型基础上进一步优化，最后对串行的多个模型做累加集成。参考：https://www.cnblogs.com/zongfa/p/9314440.html
        - ada boost的串行优化体现在：前一个模型会根据误差率更新错分样本权重，从而影响串行下一个模型的选择;
        - 而GBDT的串行优化体现在：
        """
        pass
    
    def predict_single(self):
        pass
