#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:16:38 2019

@author: ubuntu
"""
import numpy as np


def label_transform(labels, label_transform_dict={1:1, -1:0, 0:0}):
    """默认不改变label的取值范围，但可以通过该函数修改labels的对应范围
    例如svm需要label为[-1,1]，则可修改该函数。
    """
    new_labels = np.zeros(labels.shape)
    for i, label in enumerate(labels):
        new_label = label_transform_dict.get(label, label) # 获取标签，如果是dict里边有的就替换，否则保持原样
        new_labels[i] = int(new_label)   # 比如{1:1, 0:-1}就是要把1变为1, 0变为-1
    return new_labels
        

def label_to_onehot(labels):
    """标签转换为独热编码：输入的labels需要是从0开始的整数，比如[0,1,2,...]
    输出的独热编码为[[1,0,0,...],
                  [0,1,0,...],
                  [0,0,1,...]]  分别代表0/1/2的独热编码
    """
    assert labels.ndim ==1, 'labels should be 1-dim array.'
    labels = labels.astype(np.int8)
    n_col = int(np.max(labels) + 1)   # 独热编码列数，这里可以额外增加列数，填0即可，默认是最少列数
    one_hot = np.zeros((labels.shape[0], n_col))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot  # (n_samples, n_col)


def onehot_to_label(one_hot_labels):
    """把独热编码变回0-k的数字编码"""
    labels = np.argmax(one_hot_labels, axis=1)  # 提取最大值1所在列即原始从0开始的标签
    return labels


if __name__ == "__main__":
    labels = np.array([[0,1,0],[0,0,1]])
    new_labels = onehot_to_label(labels)