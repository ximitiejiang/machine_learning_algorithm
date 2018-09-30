#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:05:16 2018

@author: suliang
"""


# 这个one hot encoding的原理是什么？
import numpy as np
labels_dense = np.array([1,3,9,1,3,4,0])

num_classes = 4

num_labels = labels_dense.shape[0]  # 计算有多少个值
index_offset = np.arange(num_labels) * num_classes
labels_one_hot = np.zeros((num_labels, num_classes))
labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

#return labels_one_hot

