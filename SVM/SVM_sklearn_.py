#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:04:57 2018

@author: suliang
"""

from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X,y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.3)

# 定义模型，拟合模型
clf = svm.SVC(C=1.0, kernel='linear') 
clf.fit(X,y)
# 获得支持向量
svs = clf.support_vectors_   # 获得支持向量

# 可视化支持向量
plt.figure(figsize=(6,4),dpi=80)
plt.title('sklearn SVC')
plt.scatter(X[:,0],X[:,1],c=y*30+60)
plt.scatter(svs[:,0],svs[:,1], c='r',marker='o',s= 5,linewidths=20)

# 预测新样本
print('the point class is: {}'.format(clf.predict([[2., 2.]])))  # 预测


#---------使用自己的SVM运行-------------
import libSVM

alphas, b = libSVM.SMOsimple(X, y, 1.0, 0.01, 40)

svlist = []
for i in range(len(alphas)):
    if alphas[i]>0:
        svlist.append(i)

colour = [c*30+60 for c in y]
plt.figure(figsize = (6,4), dpi=80)
plt.title('my libSVM')
plt.scatter(X[:,0],X[:,1],c=colour)
for i in range(len(svlist)):
    plt.scatter(X[svlist[i],0],X[svlist[i],1], c='r',s=5,linewidths=20,marker='o')        

