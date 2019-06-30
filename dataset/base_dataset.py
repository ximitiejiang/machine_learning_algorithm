#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019
@author: ubuntu
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import numpy as np

class BaseDataset():
    
    def __init__(self, 
                 norm=None, 
                 label_transform_dict=None, 
                 one_hot=None):
        """数据集基类，默认支持如下变换：
        - 归一化： norm=True
        - 标签值变换: label_transform_dict = {1:1, 0:-1}
        - 标签独热编码: one_hot=True
        """
        self.dataset = self.get_dataset()
        self.datas = self.dataset.get('data', [])    # (n_sample, n_feat)
        self.labels = self.dataset.get('target', []) # (n_sample,)
        
        self.label_names = self.dataset.get('target_names', None)
        self.imgs = self.dataset.get('images', None)
        self.feat_names = self.dataset.get('feature_names', None)
        
        self.classes = set(self.labels)
        self.num_classes = len(self.classes)
        self.num_features = self.datas.shape[1]  # 避免有的数据集没有feat_names这个字段

        if norm:
            self.feats = scale(self.feats)
        if label_transform_dict:
            self.label_transform(label_transform_dict)
        if one_hot:
            self.label_to_one_hot()
            
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

