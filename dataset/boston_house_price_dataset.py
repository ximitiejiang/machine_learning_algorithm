#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import load_boston

class BostonHousePriceDataset():
    """sklearn自带波士顿房价数据集(506,13)，属于回归数据集，共计506个样本，每个样本有13个特征
    CRIM: 城镇人均犯罪率
    ZN: 城镇超过25000平方英尺的住宅区域占地比例
    INDUS: 城镇非零售用地占地比例
    CHAS: 是否靠近河边，1为靠近
    NOX: 一氧化氮浓度
    RM: 每套房产平均房间个数
    AGE: 在1940年前就盖好的且业主自住的房子比例
    DIS: 与波士顿市中心的距离
    RAD: 周边告诉公道的便利性指数
    TAX: 没10000美元的财产税率
    PTRATIO: 小学老师比例
    B: 城镇黑人比例
    LSTAT: 地位较低的人口比例

    boston_house_price.keys() = ['data','target','feature_names', 'DESCR','filename']
    其中data(506,13), target(506,), 其中target为每个房价的数值
    """
    def __init__(self):
        self.type = 'reg'    # 定义数据集类型为回归类数据集
        self.dataset = load_boston()
        self.datas = self.dataset.data      # 
        self.labels = self.dataset.target   # 
        self.feat_names = self.dataset.feature_names  # 
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]
        
        return [data, label]
    

if __name__ == '__main__':
    bost = BostonHousePriceDataset()
    data, label = bost[10]                     
    
               