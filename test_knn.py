#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""


import matplotlib.pyplot as plt
from dataset.digits_dataset import DigitsDataset
from core.knn_lib import KNN

if __name__ == "__main__":

    # get dataset
    dataset = DigitsDataset(data_type = 'train')
    # get model
    knn = KNN(dataset.datas, dataset.labels)
    # get sample
    sample_id = 1517
    sample, label, label_raw = dataset[sample_id]  # 用第2000个样本做测试
    # test and show
    pred = knn.classify(sample, k=13)
    print("the sample label is %d (raw label is %d), predict is %d"%(label, label_raw, pred))   
    plt.figure()
    plt.subplot(1,1,1)
    plt.imshow(sample.reshape(8,8), cmap='gray')