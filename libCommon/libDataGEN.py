#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 20:10:25 2018

@author: suliang
"""
import numpy as np
import matplotlib.pyplot as plt


# 分类数据创建器
def classifyData_1():
    from sklearn.datasets.samples_generator import make_classification
    X,labels=make_classification(n_samples=200,n_features=2,n_redundant=0,
                                 n_informative=2, random_state=1,
                                 n_clusters_per_class=2)
    rng=np.random.RandomState(2)
    X+=2*rng.uniform(size=X.shape)
    return X, labels
    

# 生成圆形分布的分类数据
def classifyData_2():
    from sklearn.datasets.samples_generator import make_circles
    X,labels=make_circles(n_samples=200,noise=0.2,factor=0.2,random_state=1)
    return X, labels


# 回归数据创建器        
def regressionData():
    from sklearn.datasets import make_regression
    X,y,coef=make_regression(n_samples=1000,n_features=1,noise=10,coef=True)
    #关键参数有n_samples（生成样本数），n_features（样本特征数）
    # noise（样本随机噪音）和coef（是否返回回归系数)
    # X为样本特征，y为样本输出， coef为回归系数，共1000个样本，每个样本1个特征
    return X, y, coef


# 聚类数据创建器
def clusterData_1():
    from sklearn.datasets.samples_generator import make_blobs
    center=[[1,1],[-1,-1],[1,-1]]  # 指定三个分类的中心点
    cluster_std=0.3          # 指定分类数据的标准差，标准差越大越分散
    X,labels = make_blobs(n_samples=200,centers=center,n_features=2,
                       cluster_std=cluster_std,random_state=0) 
    # 特征数=2个，随机状态？
    return X, labels    


# 分类数据集：iris莺尾花数据集导入
def irisData():
    from sklearn.datasets import load_iris
    iris = load_iris()  # 导入后可用： iris.data, iris.target, 
                      # iris.DESCR, iris_feature_names, iris_target_names
    return iris.data, iris.target
    

# 分类数据集：简化版手写数字数据集导入
def digitsData():
    from sklearn.datasets import load_digits
    digits=load_digits()
    
    return digits.data, digits.target, digits.images

# 回归数据集：boston房价数据导入, 包含13个特征，和一个labels目标房价
def bostonData():
    from sklearn.datasets import load_boston
    boston = load_boston()
    
    return boston.data, boston.target
    
# 回归数据集：糖尿病数据集,包括10个特征，和一个labels实测目标值
def diabetesData():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    
    return diabetes.data, diabetes.target

# 多元回归数据集：linnerud, 包括3个特征，和3个目标值向量
def linnerudData():
    from sklearn.datasets import load_linnerud
    linnerud = load_linnerud()
    
    return linnerud.data, linnerud.target
    

# 数据可视化
def plotCurve(X,y,*args):
    plt.scatter(X,y,c='b')
    if args:
        plt.plot(X, X*args, c='r')


# -----运行区----------
#X, labels = classifyData_2()
#X,labels = irisData()
X, labels =linnerudData()        
#X,labels,images = digitsData()
plotCurve(X[:,0],X[:,1])
#X,y,coef = regressionData()
#plotCurve(X,y,coef)

