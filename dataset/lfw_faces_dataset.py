#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019
@author: ubuntu
"""

from sklearn.datasets import fetch_lfw_people, fetch_lfw_pairs
from .base_dataset import BaseDataset
import matplotlib.pyplot as plt

class LFWPeopleDataset(BaseDataset):
    """sklearn自带人脸识别数据集LFW people(LFW=labeled faces in the wild)，数据集会自动下载到本地/home/ubuntu/scikit_learn_data
    数据集结构：属于多分类数据集，包含['data','images','target','target_names','DESCR']4种数据，
    其中data是(13233, 1914)即展平的图片像素共计13233张，而images是(13233, 62, 47)彩色图片 
    labels是(13233,)包含5749个独立的人的类别(0-5748)，里边有的人只有1张图片，有的人有上百张图片
        
    lfw_faces.keys() = ['data','images','target','target_names','DESCR']
    """
    def __init__(self, min_faces_per_person=0):
        self.min_faces_per_person = min_faces_per_person
        super().__init__()
    
    def get_dataset(self):
        return fetch_lfw_people(min_faces_per_person=self.min_faces_per_person)
    
 
class LFWPairsDataset(BaseDataset):
    """sklearn自带人脸识别配对数据集LFW pairs(LFW=labeled faces in the wild)，数据集会自动下载到本地/home/ubuntu/scikit_learn_data
    数据集结构：属于多分类数据集，包含['data','pairs','target','target_names','DESCR']4种数据，
    其中data是(2200, 5828)代表两张展平的图片像素即62*47 + 62*47 = 5828，共计2200对图片，每张图片大小是(62, 47)，
    而pairs是(2200, 2, 62, 47)代表真正的每2张图片放在一起
    labels是(2200,)表示的是(0或1),其中1表示same person, 0表示different persons
        
    lfw_faces.keys() = ['data','images','target','target_names','DESCR']
    """
    def __init__(self):
        super().__init__()
        self.pairs = self.dataset['pairs']
    
    def get_dataset(self):
        return fetch_lfw_pairs()
    
    def show(self, idx):
        """在这里是并排显示一对两张图片
        """
        if self.pairs is not None:
            img1 = self.pairs[idx][0]
            img2 = self.pairs[idx][1]
            plt.subplot(1,2,1)
            plt.imshow(img1)
            plt.subplot(1,2,2)
            plt.imshow(img2)
        else:
            print('no paris imgs can be shown.')


