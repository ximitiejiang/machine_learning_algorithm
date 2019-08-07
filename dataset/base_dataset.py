#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019
@author: ubuntu
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd

class BaseDataset():
    
    def __init__(self, 
                 norm=None, 
                 label_transform_dict=None, 
                 one_hot=None,
                 binary=None,
                 shuffle=None):
        """数据集基类，默认先通过get_dataset()获得self.dataset，里边基于sklearn的结构
        应包含["data", "target", "target_names", "images", "feature_names"]，然后在
        基类中读取这些数据存放到相应self.xx变量中，但名称做了微调来满足习惯：
        [self.datas, self.labels, self.label_names, self.imgs, self.feat_names, self.classes, self.num_classes, self.num_features]
        
        数据集通过继承基类可获得支持如下变换：
        - 归一化： norm=True
        - 标签值变换: label_transform_dict = {1:1, 0:-1}
        - 标签独热编码: one_hot=True
        - 二分类化：binary=True
        
        自定义数据集如果要继承基类的示例：
        1. 编写get_dataset()方法，里边return一个字典{"data":data, "target":target, "target_names":target_names, "feature_names":feature_names,...}
        2. super().__init__(norm=norm, label_transform_dict=label_transform_dict, one_hot=one_hot, binary=binary, shuffle=shuffle)
        3. 最终自定义数据集的接口变量包括：[self.datas, self.labels, self.label_names, self.imgs, self.feat_names, self.classes, self.num_classes, self.num_features]
        """
        # 1.提取数据
        self.dataset = self.get_dataset()
        self.datas = self.dataset.get('data', [])    # (n_sample, n_feat)
        self.labels = self.dataset.get('target', []) # (n_sample,)        
        self.label_names = self.dataset.get('target_names', None)
        self.imgs = self.dataset.get('images', None)
        self.feat_names = self.dataset.get('feature_names', None)
        # 2.扩展变量 
        self.classes = set(self.labels)
        self.num_classes = len(self.classes)
        self.num_features = self.datas.shape[1]  # 避免有的数据集没有feat_names这个字段
        # 3.数据变换
        if shuffle:
            idx = np.random.permutation(self.labels)
            self.datas = self.datas[idx]
            self.labels = self.labels[idx]
            self.imgs = self.imgs[idx] if self.imgs is not None else None
        if norm:
            self.datas = scale(self.datas)
        if label_transform_dict:
            self.label_transform(label_transform_dict)
        if one_hot:
            self.label_to_one_hot()
        if binary:
            self.get_binary_dataset()

    def label_transform(self, label_transform_dict):
        """默认不改变label的取值范围，但可以通过该函数修改labels的对应范围
        例如svm需要label为[-1,1]，则可修改该函数。
        """
        if label_transform_dict is None:
            pass
        else:  # 如果指定了标签变换dict
            self.labels = np.array(self.labels).reshape(-1)  #确保mat格式会转换成array才能操作
            assert isinstance(label_transform_dict, dict), 'the label_transform_dict should be a dict.' 
            for i, label in enumerate(self.labels):
                new_label = label_transform_dict[label]
                self.labels[i] = int(new_label)   # 比如{1:1, 0:-1}就是要把1变为1, 0变为-1
    
    def label_to_one_hot(self):
        """标签转换为独热编码：输入的labels需要是从0开始的整数，比如[0,1,2,...]
        输出的独热编码为[[1,0,0,...],
                      [0,1,0,...],
                      [0,0,1,...]]  分别代表0/1/2的独热编码
        """
        assert self.labels.ndim ==1, 'labels should be 1-dim array.'
        n_col = np.max(self.labels) + 1   # 独热编码列数，这里可以额外增加列数，填0即可，默认是最少列数
        one_hot = np.zeros((self.labels.shape[0], n_col))
        one_hot[np.arange(self.labels.shape[0]), self.labels] = 1
        self.labels = one_hot  # (n_samples, n_col)
    
    def get_binary_dataset(self):
        """从原多分类数据集随机提取其中两类得到二分类数据集"""
        label_unique = np.unique(self.labels)
        random_labels = np.random.permutation(label_unique)[:2]  # 提取前2个标签
        idx = (self.labels == random_labels[0]) | (self.labels == random_labels[1])  # [False, True,..] or [False, True,..] -> [False, True,..]
        
        labels_binary = self.labels[idx]  # 筛选出其中2个类的labels
        feats_binary = self.datas[idx]   # 筛选出其中2个类的feats
        # 转换数据集        
        self.datas = feats_binary
        self.labels = labels_binary
        self.label_names = random_labels        
    
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
        
        # 绘制类别统计结果
        feat_names = self.feat_names if self.feat_names else np.arange(self.datas.shape[1])
        df_data = pd.DataFrame(self.datas, columms=feat_names)
        grr = pd.scatter_matrix(df_data, c=self.labels, 
                                figsize=(15,15),
                                marker='o',
                                hist_kwds={'bins':20}, s=60, alpha=0.8)
    
    def show(self, idx):
        """用于显示图片样本"""
        if self.imgs is not None:
            img = self.imgs[idx]
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            print('no imgs can be shown.')

