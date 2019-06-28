#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019

@author: ubuntu
"""

import numpy as np
import pandas as pd
from .base_dataset import BaseDataset

class PimaIndiansDiabetesDataset(BaseDataset):
    """第三方印第安人糖尿病数据集，共计768个样本，每个样本有8个特征，标签labels为(0,1)。
    属于多特征二分类数据集。

    其中data(768, 8), target(768,)
    """
    def __init__(self, path='./dataset/simple/pima_indians_diabetes.csv', 
                 norm=None, label_transform_dict=None):
        self.path = path
        super().__init__(norm=norm, label_transform_dict=label_transform_dict)
    
    def get_dataset(self):
        raw = np.array(pd.read_csv(self.path))
        dataset = dict(data = raw[:, :-1], target = raw[:, -1])
        return dataset
    
            