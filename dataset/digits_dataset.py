#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import load_digits

class DigitsDataset():
    """sklearn自带手写数字集(1797,64)共计1797个样本
    digits.keys() = ['images','data','target_names','DESCR','target']
    其中imgaes为(8,8)ndarray, data为(1797,64)ndarray展开的图片值， target为(1797,)标签值
    
    """
    def __init__(self):
        """固定接口为self.datas, self.labels"""
        self.dataset = load_digits()        # dict, len=5
        self.datas = self.dataset['data']   # (1797,64)
        self.labels = self.dataset['target']  # (1797,)
        self.imgs = self.dataset['images']

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        img = self.datas[idx]
        label = self.labels[idx]
        
        return [img, label] 