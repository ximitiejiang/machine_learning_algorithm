#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 10:00:32 2018

包含的重要子程序包括：
    - modelsFit(x_train, y_train): 用来循环拟合所有sklearn的模型
    - learningCurve(X,y,model): 用来拟合和绘制学习曲线
    
@author: suliang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# 这个numpy自带的求分位数的函数计算出来跟浙大版《概率论与数理统计》的计算方法不同
def quantile0(arr):
    d50 = np.median(arr)
    d25 = np.percentile(arr, 25)
    d75 = np.percentile(arr, 75)
    return [d25, d50, d75]


# 自己编写的计算分位数的函数，基于浙大版《概率论与数理统计》的计算方法
def quantile1(arr):
    d = [0,0,0]
    n = arr.shape[0]   # 数组个数
    tmp = [0.25, 0.50, 0.75]  
    for i, p in enumerate(tmp): # 循环判断和计算三个分位数
        if math.modf(n*p)[0]==0:  # 判断是否为整数不能直接判断int，因为浮点数运算结果仍然是浮点数
            k = int(n*p)          # 解决办法是判断小数为是否为0
            d[i] = 0.5*(arr[k-1] + arr[k]) # 整数情况： 取前后两数平均
        else:
            k = int(n*p)            # 非整数情况：下取整
            d[i] = arr[k]
    return d


# 绘制箱线图和异常点：待完成    
def box0(df):
    p = df.boxplot(return_type = 'dict') # 对dataframe格式绘制箱式图，有几列就绘制几个，如果不指定return_type在后边异常值处理会报错
    # 接下来标准出异常值位置:这里只标出了第一列
    x = p['fliers'][1].get_xdata()    # flies 为异常标签  [1]代表是第一列
    y = p['fliers'][1].get_ydata()    # x为?，y为异常值集合
    y.sort()
    print(x,y)
    for i in range(len(x)):
        if i >0:
            plt.annotate(y[i], xy = (x[i],y[i]), xytext = (x[i]+0.05 - 0.8/(y[i]-y[i-1]),y[i])) # 添加注释，显示位置微调
        else:
            plt.annotate(y[i], xy = (x[i],y[i]), xytext = (x[i]+0.08,y[i])) 
    plt.show()


# 数据预处理
def preProcessing(data):

    # 空值检测
    print('-'*10+'original null: \n{}'.format((data.isnull()).sum()))
    data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean())
    data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0])
    print('-'*10+'new null: \n{}'.format((data.isnull()).sum()))
    
    # 0值检测
    print('original visibility zero num: {}'.format((data['Item_Visibility']==0).sum()))
    data.loc[data['Item_Visibility'] == 0, 'Item_Visibility'] = data['Item_Visibility'].mean()
    print('new visibility zero num: {}'.format((data['Item_Visibility']==0).sum()))
    
    return data


def abnomalPointsProcessing():
    pass



    
    # 主成分分析PCA

  


# 单个模型调参
def modelParameterJustifier():
    pass



    
    
# ---------------测试程序----------------------------    
def test_featShrink():
    input = pd.read_csv('Train_big_mart_III.csv')
    input = preProcessing(input)
    train = input.iloc[:,0:-1]
    label = input.iloc[:,-1]
    
    featShrink(train, label)


def test1():
    a = np.array([1,2,3,4,5,6,7,8,9,10])
    b = np.array([122,126,133,140,145,145,149,150,157,
                  162,166,175,177,177,183,188,199,212])
    c = np.array([102,110,117,118,122,123,132,150])

    d0 = quantile0(b)
    print(f'standard data = {d0}')

    d1 = quantile1(b)
    print(f'calculate data = {d1}')
