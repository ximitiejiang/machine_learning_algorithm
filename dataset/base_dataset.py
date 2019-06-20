#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019
@author: ubuntu
"""

import matplotlib.pyplot as plt

class BaseDataset():
    
    def __init__(self):
        
        self.dataset = self.get_dataset()
        self.datas = self.dataset.get('data', [])    # (n_sample, n_feat)
        self.labels = self.dataset.get('target', []) # (n_sample,)
        
        self.label_names = self.dataset.get('target_names', None)
        self.imgs = self.dataset.get('images', None)
        self.feat_names = self.dataset.get('feature_names', None)
        
        self.classes = set(self.labels)
        self.num_classes = len(self.classes)
        self.num_features = self.datas.shape[1]  # 避免有的数据集没有feat_names这个字段
        
        
        
    def get_dataset(self):
        raise NotImplementedError('the get_dataset function is not implemented.')
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]
        
        return [data, label]
    
    def statistics(self):
        """用来统计各个类别样本个数是否平衡"""
        class_num_dict = {}
        for label in self.labels:
            class_num_dict[label] = class_num_dict.get(label, 0) + 1
        
        # 打印统计结果
        for key, value in sorted(class_num_dict.items()):
            print('class %d: %d' % (key, value))
        print('total num_classes: %d'%self.num_classes)
        
        # 绘制二维数据的分布图
        if self.num_features == 2:
            color = [c*64 + 128 for c in self.labels.reshape(-1)]
            plt.scatter(self.datas[:,0], self.datas[:,1], c=color)
        
        # 绘制类别统计结果图片
#        plt.subplot(1,1,1)
#        plt.hist()
    
    def show(self, idx):
        """用于显示图片样本"""
        if self.imgs is not None:
            img = self.imgs[idx]
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            print('no imgs can be shown.')

