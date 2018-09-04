#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 20:27:50 2018

@author: suliang
"""
#


# 基础模型调用
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
    models.append(('LoR', LogisticRegression(penalty = 'l2',
                                             dual = 'True',
                                             C=1,
                                             solver = 'sag',
                                             multi_class = 'mvm',
                                             class_weight = 'balanced')))
    models.append(('DeT', DecisionTreeClassifier(max_depth=7)))
    models.append(('Bagging',BaggingClassifier(DecisionTreeClassifier(random_state=1))))
    models.append(('RanF', RandomForestClassifier(random_state=1)))
    models.append(('AdaB', AdaBoostClassifier(random_state=1)))
    models.append(('GBM', GradientBoostingClassifier(learning_rate=0.01,
                                                     random_state=1)))
    models.append(('XGB', xgb.XGBClassifier(random_state=1,
                                            learning_rate=0.01)))

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



# 学习曲线绘制：评估模型的收敛性    
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
    