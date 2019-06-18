#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019
@author: ubuntu
"""

from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

class LFWFacesDataset():
    """sklearn自带人脸识别数据集(LFW=labeled face in the wild)，数据集会自动下载到本地/home/ubuntu/scikit_learn_data
    数据集结构：属于多分类数据集，包含['data','images','target','DESCR']4种数据，
    其中data是(400,4096)即展平的图片像素共计400张，而images是(400, 64,64)即原始400张灰度图, 
    labels是(400,)包含40个类别(0-39)即40个人的脸每个人有10张图片
    
        
    lfw_faces.keys() = ['data','images','target','DESCR']
    """
    def __init__(self):
        self.dataset = fetch_lfw_people()
        self.datas = self.dataset['data']      # 
        self.labels = self.dataset['target']   # 

        self.imgs = self.dataset['images']
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]
        
        return [data, label]
    
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
        img = self.imgs[idx]
        plt.imshow(img, cmap=plt.cm.gray)

if __name__ == '__main__':
    faces = LFWFacesDataset()
    faces.statistics()
    idx = 65
    data, label = faces[idx]
    faces.show(idx)     
    print('this pic idx = %d, label = %d' % (idx, label))
