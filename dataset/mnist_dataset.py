#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:54:50 2019

@author: ubuntu
"""

import pandas as pd

class MnistDataset():
    """采用kaggle版本的mnist数据集https://www.kaggle.com/c/digit-recognizer/data
    数据被处理成train.csv和test.csv，每张图片像素为28*28, 被展平为一行784个像素
    其中train.csv为(42000,785), 即42000个样本，且第一列为label(0-9), 剩下为像素(0-255)
    其中test.csv为(28000, 784), 即28000个样本
    另一个外部转换过的train_binary.cvs数据集是把tarin.csv的label列进行转换，原来label=0不变，原来label>0的改为1
    从而变成一个二分类数据集(两个类别是0或非零)
    """
    def __init__(self, root_path='./mnist/', data_type='train'):
        """固定接口为self.datas, self.labels"""
        train_path = root_path + 'train.csv'
        test_path = root_path + 'test.csv'
        train_binary_path = root_path + 'train_binary.csv'
        
        if data_type == 'train':
            path = train_path
        elif data_type == 'train_binary':
            path = train_binary_path
        elif data_type == 'test':
            path = test_path
        else:
            raise ValueError('wrong data type, only support train/train_binary/test.')
              
        dataset = pd.read_csv(path, header=0).values  # (42000, 785)
        self.datas = dataset[:, 1:]
        self.labels = dataset[:,0]  

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        img = self.datas[idx]
        label = self.labels[idx]
        
        return [img, label] 

if __name__ == "__main__":
    mn = MnistDataset()
    data, label = mn[3]
    