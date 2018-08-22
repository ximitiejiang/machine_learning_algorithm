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


def modelsFit(x_train, y_train):    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    import xgboost as xgb
    
    from sklearn.model_selection import KFold, cross_val_score
    import datetime
    
    starttime = datetime.datetime.now()
    models = []
    models.append(('KNN', KNeighborsClassifier(n_neighbors=9)))
    models.append(('LoR', LogisticRegression()))
    models.append(('DeT', DecisionTreeClassifier(max_depth=7)))
    models.append(('Bagging',BaggingClassifier(DecisionTreeClassifier(random_state=1))))
    models.append(('RanF', RandomForestClassifier(random_state=1)))
    models.append(('AdaB', AdaBoostClassifier(random_state=1)))
    models.append(('GBM', GradientBoostingClassifier(learning_rate=0.01,random_state=1)))
    #models.append(('XGB', xgb.XGBClassifier(random_state=1,learning_rate=0.01)))

    results = []
    for name, model in models:
        kf = KFold(n_splits = 5) # 如果要打乱，则需要固定随机，也就是同时定义shuffle=true, random_state=0
        cv_score = cross_val_score(model, x_train, y_train, cv=kf)
        results.append((name, cv_score))
        print('......{} fit finish'.format(name))

    # 显示结果    
    for i in range(len(results)):
        print('Model name: {}; CV score: {}'.format(results[i][0], results[i][1].mean()))

    lasttime = datetime.datetime.now() - starttime
    print('fit finish! last time = {}s'.format(lasttime.seconds))
    return models, results


def learningCurve(X,y,model):
    from sklearn.model_selection import learning_curve
    import numpy as np
    plt.figure(figsize=(6,5), dpi = 80) # figsize定义的是(width, height),千万不要理解成行数列数，否则就编程高，宽了。
    plt.title('Learning Curve (degrees ={},penalty={})'.format(degrees[1], penalty[0]))
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim([0.8,1.01])
    from sklearn.model_selection import ShuffleSplit

    # 核心语句1: 交叉验证生成器
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) # 交叉验证生成器
    # 核心语句2: 学习曲线参数生成器
    train_sizes, train_scores, test_scores = \
            learning_curve(model,   # 带参模型输入
                           X, y,    # 完整数据集数据输入
                           cv=cv,   # 交叉验证生成器用ShuffleSplit
                           train_sizes=np.linspace(.1, 1.0, 5))# 取数据时的size相对位置

    train_scores_mean  = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    plt.grid()
    plt.legend(loc="best")

# ---------------测试程序----------------------------    
def test():
    a = np.array([1,2,3,4,5,6,7,8,9,10])
    b = np.array([122,126,133,140,145,145,149,150,157,
                  162,166,175,177,177,183,188,199,212])
    c = np.array([102,110,117,118,122,123,132,150])

    d0 = quantile0(b)
    print(f'standard data = {d0}')

    d1 = quantile1(b)
    print(f'calculate data = {d1}')
