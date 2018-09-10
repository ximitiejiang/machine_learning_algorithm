#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 20:29:24 2018

@author: suliang

Data dimension reduction特征尺寸缩减，讨论的是特征降维的方法：

特征降维的目的：对于有的数据，虽然维度很高，但大部分的方差都包含在少数特征里边。
这时通过特征降维就可以保留少数方差占比高的特征而丢弃方差占比低的特征，从而实现特征降维。

方差分析：通过做源数据的方差分析和排序，也可看到前n个特征占据的总方差百比分很大，舍弃的特征对方差基本无贡献

特征降维的伪代码逻辑：先数据去除平均值，然后求取协方差矩阵，得到协方差矩阵的特征值和特征向量，
保留特征值最大的前n个特征向量，用特征向量乘以源数据就得到降维后的特征。

常用的特征降维的手段
(1) PCA特征降维
(2)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clusterData_1():
    from sklearn.datasets.samples_generator import make_blobs
    center=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]]  # 指定三个分类的中心点
    cluster_std=[0.2, 0.1, 0.2, 0.2]          # 指定分类数据的标准差，标准差越大越分散
    X,labels = make_blobs(n_samples=10000, centers=center,n_features=3,
                       cluster_std=cluster_std,random_state=9) 

    return X, labels 


# 特征降维，PCA方法
def PCAMethod(X, n):
    from sklearn.decomposition import PCA
    ''' 主成分模型参数：
        * n_components, 要保留的主成分矢量个数，如果没设置则取默认值min(n_samples, n_features)
          如果='mle'则自动选择特征个数
          如果=数值，则该数值为0-1之间的数值，代表保留的所有主成分方差之和占总方差的最小比例
        * copy, 默认是True, 是否复制原始数据进行主成分生成，也可直接覆盖
        * svd_solver, 默认是'auto',用来指定奇异值分解SVD的方法
          包括'auto, 'full', 'arpack', 'randomized'，其中randomized适用大数据多维度，full是传统SVD
        * whiten, 默认是False, 是否白化，使每个特征具有相同方差
    '''
    pca = PCA(n_components=n,
              copy = True,
              svd_solver='auto')
    X_new = pca.fit_transform(X) # 一步完成fit和transform
    print('variance = ', pca.explained_variance_)
    print('variance ratio = ', pca.explained_variance_ratio_)
    return X_new

# 特征降维，

# -------------------调试区------------------------------------------------    
if __name__ == '__main__':
    
    test_id = 1
    
    if test_id == 0:
        data, label = clusterData_1()
        plt.scatter(data[:,0], data[:,1], c = label*30+30)
        result = PCAMethod(data,3)    # 查看在n=3时的方差占比：第一个特征占比98.3%
    
    elif test_id == 1:
        data, label = clusterData_1()
        result = PCAMethod(data,2)    # 查看在n=2时的方差占比：第一个特征占比98.3%
        plt.scatter(result[:,0], result[:,1], c=label*30+30)  
        # 数值上2个主成分占方差98%以上，图形上2个主成分可以把4个类完全区分表示
        
    else:
        print('Wrong test_id!')
        
        
        
        
        
        
        
        