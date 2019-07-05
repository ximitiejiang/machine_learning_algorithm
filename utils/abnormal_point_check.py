#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:07:42 2019

@author: ubuntu
"""
import numpy as np
import seaborn as sns

class AbnormalPointCheck():
    
    def __init__(self, data):
        """异常点/离群点检测：通常用于对某一列的特征进行检测
        
        方式1：计算标准差，在3倍标准差之外的点就是离群点(常规正态分布数据99.7%分布在3倍标准差范围以内)
        方式2：绘制箱型图(也叫五数概括法)，中间是中位数median,箱子上下边沿是第一四分位数Q1和第三四分位数Q3，
               一个箱子的长度就是IQR,从Q1到下边沿距离是1.5IQR，从Q3到上边沿的距离是1.5IQR,上下边沿之外的就是离群点
        """
        self.data = data
    
    def find(self):
        sns.boxplot(data=self.data)

if __name__ == "__main__":
    data = np.random.randn(50000) *20 + 20
    sns.boxplot(data=data)  # 绘制箱型图，发现-35到75之外的点为离群点

        