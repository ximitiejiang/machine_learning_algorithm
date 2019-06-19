#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

from sklearn.datasets import fetch_olivetti_faces
from .base_dataset import BaseDataset

class OlivettiFacesDataset(BaseDataset):
    """sklearn自带人脸识别数据集，数据集会自动下载到本地/home/ubuntu/scikit_learn_data
    数据集结构：属于多分类数据集，包含['data','images','target','DESCR']4种数据，
    其中data是(400,4096)即展平的图片像素共计400张，而images是(400, 64,64)即原始400张灰度图, 
    labels是(400,)包含40个类别(0-39)即40个人的脸每个人有10张图片
    
        
    olivetti_faces.keys() = ['data','images','target','DESCR']
    """
    def __init__(self):
        super().__init__()
    
    def get_dataset(self):
        return fetch_olivetti_faces()


                
              