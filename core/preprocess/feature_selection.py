#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:44:09 2019

@author: ubuntu
"""
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def find_k(feats, k_list, required_ratio = 0.95):
    """通过PCA降维可以从特征中寻找代表性最强的特征而抛弃代表性不强的特征
    1.做pca降维的目的：例如一张图片大小64*64, 像素展平就相当于每个样本有4096个特征, 
    特征个数太多了不仅影响训练速度，而且很多时候样本个数都没有特征个数多，此时完全无法进行机器学习算法训练。
    2. pca的原理：
        
    3.选择多少个特征，取决于数据还原率的要求，比如数据还原率为95%则需要保留前k1个重要特征
    而还原率85%对应k2个特征, 所以只要定义了数据还原率就能通过pca计算出需要哪些特征
    4. pca特征降维与图片尺寸resize操作的区别：？？？
    (参考：scikit-learn机器学习)
    Args:
        k_list(list): 表示k的取值比如range(10,300,10)
        
    Returns:
        explained_variance_ratio
    """
    explained_variance_ratio = []
    for k in k_list:
        pca = PCA(n_components = k)
        X_pca = pca.fit_transform(feats)
        explained_variance_ratio.append(np.sum(pca.explained_variance_ratio_))   # 计算出在该k值下的数据还原率(0-100%)
    plt.subplot(1,1,1)
    plt.title('explained variance ratio for PCA')
    plt.plot(k_list, explained_variance_ratio)
    
    # get k on required_ratio
    
    return explained_variance_ratio, ratio

if __name__ == "__main__":
    from sklearn.datasets import fetch_olivetti_faces
    data = fetch_olivetti_faces().data    
    label = fetch_olivetti_faces().target
    
    k_list = np.arange(10, 300, 10) # 代表k从10到300,间隔10
    ratio = find_k(data, k_list)