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