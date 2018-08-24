#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 19:40:18 2018

@author: suliang
"""


'''
Hardware: 1.3Ghz intel i5, 8G DDR3

>dataimport finish! last time = 32s
>featureing engineering finish! last time = 574s
>fit finish! last time = 490s
>predict finish! last time = 3s
>writing to document finish! last time = 0s
>ALL Finished!!!.......

对于1000个样本，5折交叉验证跑下来花了10分钟，其中8分钟花在GBM上
Model name: KNN; CV score: 0.185
Model name: LoR; CV score: 0.512
Model name: DeT; CV score: 0.202
Model name: Bagging; CV score: 0.43600000000000005
Model name: RanF; CV score: 0.358
Model name: AdaB; CV score: 0.14200000000000002
Model name: GBM; CV score: 0.445
fit finish! last time = 636s

问题：
1. 为什么KNN这么差？KNN的缺陷是什么？
2. 为什么DeT这么差（所以导致RanF也很差）？DeT的缺陷是什么？
3. 
'''

# 题目定义为多分类问题
# v1版程序改进：先导入10,000个样本（原数据集是100,000个），先用这1/10的样本跑算法
# 方便现有的硬件可以跑得更快（原数据集跑一个逻辑回归分类算法需要18分钟）。
# 然后循环尝试所有算法，找到得分最高的算法。
# 最后用得分最高的算法跑一次完整的数据



def importData(row_num):
    import pandas as pd
    import datetime
    # 导入数据: 改为只导入10,000个数据（10%）
    dataAddr = ''
    starttime = datetime.datetime.now()
    df_train = pd.read_csv('train_set.csv', nrows=row_num)
    df_train.drop(columns = ['article', 'id'], inplace = True)
    
    df_test = pd.read_csv('test_set.csv', nrows=row_num)
    df_test.drop(columns = ['article'], inplace = True)
    
    lasttime = datetime.datetime.now() - starttime
    print('import finish! last time = {}s'.format(lasttime.seconds))
    return df_train, df_test

def importData_2():  # pd.read_csv()的循环导入方法: 该方法跟我用的直接读入指定行数什么区别？
    chunks = pd.read_csv('train_Set.csv', iterator = True)
    df_train = chunks.get_chunk(10000)
    
    chunk = []
    loop = True
    chunksize = 5000
    while loop:
        try:
            chunk = df_train.get_chunk(chunksize)
            chunk.drop
        except StopIteration:
            loop = False


def featureEngineering(df_train, df_test):
    import datetime
    from sklearn.feature_extraction.text import CountVectorizer
    # 特征工程
    starttime = datetime.datetime.now()
    vectorizer = CountVectorizer(ngram_range = (1,2), min_df = 3, 
                                 max_df = 0.9, max_features = 100000)
    vectorizer.fit(df_train['word_seg'])
    x_train = vectorizer.transform(df_train['word_seg'])
    y_train = df_train['class'] - 1
    
    x_test = vectorizer.transform(df_test['word_seg'])
    lasttime = datetime.datetime.now() - starttime
    print('featureing engineering finish! last time = {}s'.format(lasttime.seconds))
    
    return x_train, y_train, x_test

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


def paraCompare(data, labels):
    # 发现逻辑回归依然最高，那就继续逻辑回归调参
    from sklearn.linear_model import LogisticRegression
    # 引入参数选择工具
    from sklearn.model_selection import GridSearchCV
    
    parameters={'penalty':('l1','l2'),'C':[0.1,1,10,100]}
    LoR = LogisticRegression()
    clf = GridSearchCV(LoR, parameters)
    clf.fit(data, labels)
    
    print(clf.best_estimator_)



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
    

def test():
    df_train, df_test = import data(10000) # row_num = 10,000
    featureEngineering(df_train, df_test)
    
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()


'''
# 基于模型计算测试集结果
starttime = datetime.datetime.now()
y_test = logclf.predict(x_test)
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:,['id','class']]
lasttime = datetime.datetime.now() - starttime
print('predict finish! last time = {}s'.format(lasttime.seconds))


# 结果写入文件
starttime = datetime.datetime.now()
df_result.to_csv('test_set.csv', index = False)
lasttime = datetime.datetime.now() - starttime
print('writing to document finish! last time = {}s'.format(lasttime.seconds))
print('ALL Finished!!!.......')

'''

