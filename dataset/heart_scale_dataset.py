#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:54:50 2019

@author: ubuntu
"""

from .base_dataset import BaseDataset
import numpy as np

class HeartScaleDataset(BaseDataset):
    """参考SVM算法例程的数据集
    """
    def __init__(self, root_path='./dataset/simple/'):
        
        self.path = root_path + 'heart_scale'
        super().__init__() # 先准备self.path再init
    
    def get_dataset(self):
        data = []
        label = []
        with open(self.path, 'r') as f:
            for line in f.readlines():
                lines = line.strip().split(' ')
                # 提取得出label
                label.append(float(lines[0]))
                # 提取出特征，并将其放入到矩阵中
                index = 0
                tmp = []
                for i in range(1, len(lines)):
                    li = lines[i].strip().split(":")
                    if int(li[0]) - 1 == index:
                        tmp.append(float(li[1]))
                    else:
                        while(int(li[0]) - 1 > index):
                            tmp.append(0)
                            index += 1
                        tmp.append(float(li[1]))
                    index += 1
                while len(tmp) < 13:
                    tmp.append(0)
                data.append(tmp)
        
        dataset = {}
        dataset['data'] = np.array(data)
        dataset['target'] = np.array(label)
        return dataset
        