#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

class DigitsDataset():
    """sklearn自带手写数字集(1797,64)共计1797个样本
    digits.keys() = ['images','data','target_names','DESCR','target']
    其中imgaes为(8,8)ndarray, data为(1797,64)ndarray展开的图片值， target为(1797,)标签值
    
    """
    def __init__(self, data_type = 'binary'):
        """固定接口为self.datas, self.labels"""
        self.dataset = load_digits()        # dict, len=5
        self.datas = self.dataset['data']   # (1797,64)
        
        if data_type == 'binary':
            # 如果是二分类问题，则把标签数据修改：0依旧是0, 1-9改为1
            labels_binary = np.zeros((len(self.datas),))
            for idx, label in enumerate(self.dataset['target']):
                if label > 0:
                    labels_binary[idx] = 1
                else:
                    labels_binary[idx] = 0
            self.labels = labels_binary.astype(np.int8)
            self.labels_raw = self.dataset['target']
        else:
            self.labels = self.dataset['target']  # (1797,)
            self.labels_raw = self.labels                
        self.imgs = self.dataset['images']

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        img = self.datas[idx]
        label = self.labels[idx]
        label_raw = self.labels_raw[idx]
        
        return [img, label, label_raw] 
    
    def statistics(self):
        classes = set(self.labels)
        n_classes = len(classes)
        
        class_num_dict = {}
        for label in self.labels:
            class_num_dict[label] = class_num_dict.get(label, 0) + 1
        print('num_classes: %d'%n_classes)
        for key, value in sorted(class_num_dict.items()):
            print('class %d: %d' % (key, value))
    
    def show(self, idx):
        if self.imgs is not None:
            img = self.imgs[idx]
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            print('no imgs can be shown.')
            
if __name__ == "__main__":
    dg = DigitsDataset(data_type='train')
   
    dg.statistics()
    
    idx = 222
    dg.show(idx)
    img, label, label_raw = dg[idx]
    print('label = %d, label_raw = %d' % (label, label_raw))