#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:59:16 2019

@author: ubuntu
"""
import numpy as np
from .base_model import BaseModel

class KNN(BaseModel):
    def __init__(self, feats, labels, k=5):
        """ knn algorithm lib
        特点：支持二分类和多分类，无模型参数，支持线性可分和非线性可分数据
        
        knn算法通过计算待预测数据与每个样本的距离，提取距离最近的k个样本投票决定待预测样本的类别
        knn算法没有预训练过程，也没有预训练参数，所以也就没有可保存的模型，都是实时计算距离实时投票实时获得预测结果
        优点：算法简单，可处理二分类和多分类
        缺点：没有可训练的参数保存，每次都要实时计算才能进行预测
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)
        """
        super().__init__(feats, labels)
        self._k = k
        
        self.model_dict['model_name'] = 'KNN' + '_k' + str(self._k)
        self.model_dict['k'] = self._k
    
    def train(self):
        return self
    
    def predict_single(self, sample_single):
        """
        Args:
            data(numpy): (1, m_feats) or (m_feats,)，如果是(m,n)则需要展平
            k(int): number of neibours
        Returns:
            label(int)
        """
        assert isinstance(sample_single, np.ndarray), 'data should be ndarray.'
        assert (sample_single.shape[0]==1 or sample_single.ndim==1), 'data should be flatten data like(m,) or (1,m).'
        assert (self._k % 2 == 1), 'k should be odd number.'
        # calculate distance
        tiled_data = np.tile(sample_single, (self.feats.shape[0], 1))           # (n, m)把输入数据堆叠成特征的高度
        dist = np.sqrt(np.sum((tiled_data - self.feats)**2, axis=1))  # (n,) distances = np.sqrt((xi - x)^2) 
        
        # sort distance and vote
        dist_sort_index = np.argsort(dist)   #(n,)
        count_dict = {}  # 存储  {标签：个数}
        for i in range(self._k):  #找最近的k个样本 
            dist_label = self.labels[dist_sort_index[i]]   # 获得距离对应的标签
            count_dict[dist_label] = count_dict.get(dist_label, 0) + 1  # 如果有这个距离对应标签key，则个数+1, 否则在0的基础上+1  
        
        # get most counted label
        label, _ = max(zip(count_dict.keys(), count_dict.values()))   # 字典排序
        return label

