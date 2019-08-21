#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import load_digits
import numpy as np
from .base_dataset import BaseDataset

class DigitsDataset(BaseDataset):
    """sklearn自带手写数字集(1797,64)共计1797个样本
    digits.keys() = ['images','data','target_names','DESCR','target']
    其中imgaes为(8,8)ndarray, data为(1797,64)ndarray展开的图片值， target为(1797,)标签值
    
    """
    def __init__(self, 
                 norm=None, label_transform_dict=None, one_hot=None, binary=None, shuffle=None):
        """可以设置data_type = 'binary'，从而输出标签变为1(数字1-9),0(数字0)
        """
        super().__init__(norm=norm, 
                         label_transform_dict=label_transform_dict, 
                         one_hot=one_hot,
                         binary=binary,
                         shuffle=shuffle)
    
    def get_dataset(self):
        return load_digits()
    
