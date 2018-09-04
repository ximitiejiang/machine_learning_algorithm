#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 20:17:01 2018

@author: suliang

libDataPP = lib of Data Pre-Processing
数据预处理
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 一个简单导入数据子程序，用于本lib测试用
def loadData():
    import pandas as pd
    data = pd.read_csv('Train_big_mart_III.csv')
    
    train = data.iloc[:,:-1]
    label = data.iloc[:,-1]
    
    return train, label


'''------------------------预处理---------------------------------------------
1. 预处理-缺失值: 缺失值处理永远是数据处理的第一步
    * 如果是连续特征：可以用平均值，或者中位数填充
    * 如果是离散特征：可以用众数填充
    * preprocessing.Imputer

2. 预处理-特殊特征
    * 时间特征处理：1)转化成到某个时间点的时间差，从而变为一个连续特征；
      2)分解为年/月/日/小时数等多个离散特征；
      3)把时间分段加权重，转为离散特征
    
    * 地理特征处理：1)分解为城市/区县/街道等多个离散特征；
      2)转化为经纬度组合的连续特征，便于判断用户分布特征
    
    * 离散特征连续化：线性回归逻辑回归只能处理连续特征，所以需要连续化，OneHotEncoder
    * 离散特征离散化：1)文字转离散数字，可用oneHotEncoder
    
    * 对于文本编码：可以用word2vec转换
    
    
3. 预处理-标准化/归一化
    * 标准化
    * 归一化
    * 正则化

4. 预处理-异常点检测
    * 使用iForest或者one class SVM算法来过滤异常点
    * 采用聚类算法
    
5. 预处理-处理数据不平衡：训练集A，B样本如果是90%vs10%，而测试集A，B样本是50%vs50%,模型泛化能力会很差
    * 权重法：对数据进行加权，绝大多数sklearn算法都可以对class weight或sample weight进行设置
    * 
'''


# 缺失值比率判断: 去除缺失值太多的特征
def checkHighPercNullFeat(train):
    nullPerc = train.isnull().sum()/len(train)*100  # 计算每个变量缺失值比例
    dropindex = np.where(nullPerc >= 20)
    for i in dropindex[0]:
        train = train.drop(train.columns[i], axis = 1)  # 更新train
    return train

# 缺失值填充
def fillNull():
    pass


'''
标准化：也叫z-score, 变为均值为0，方差为1的高斯标准正态分布
标准化方法：xhat = (x - mu)/theta (减均值，除方差)
标准化特点：不会改变分布，对较大的样本比较合适
'''
def dataStandard(X):   #
    from sklearn import preprocessing
    import numpy as np
    
    X = np.array([[ 1., -1.,  200.],
                 [ 2.,  0.,  100.],
                 [ 0.,  1., -100.]])
    X_scaled = preprocessing.scale(X_train)

    X_scaled.mean(axis=0)
    X_scaled.std(axis=0)
    
    return X_scaled
    

'''
归一化：也叫min-max normalization
归一化方法：xhat = (x - min)/(max - min)
归一化特点：改变不同特征的维度到(0,1)之间，距离/分布都变了，优点是提高迭代速度和精度，缺点是容易受异常点影响，鲁棒性差
'''
def dataNormalize():
    from sklearn import preprocessing
    import numpy as np
    X = np.array([[ 1., -1.,  200.],
                 [ 2.,  0.,  100.],
                 [ 0.,  1., -100.]])
    X_normalized = preprocessing.normalize(X, norm='l2')
    
    return X_normalized


'''
数据编码: one-hot独热编码
'''
def oneHotEncode():
    pass

def dummyEncode():
    pass

# 特征扩展成多项式特征
def d():
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    poly.fit_transform(X)

   

'''---------------------------特征选择---------------------------------------
特征选择：这里包括去掉多余特征，增加新特征
1. 过滤法：去除特征
    * 方差过滤
    * 相关系数过滤
    * 卡方检验检测相关性
    * 互信息：从决策树的信息熵角度评价相关性(mutual_info_classif(分类)和mutual_info_regression(回归))

2. 嵌入法：去除特征
    * 采用正则化：通常采用逻辑回归加正则化来筛除容易变为0的特征列
    * 也可用决策树或GBDT来筛选，只要可以得到特征系数或特征重要度的算法都可以用于嵌入法的基学习器

3. 包装法：去除特征
    * 递归消除特征法(RFE)：比如SVM-RFE算法，即用SVM来作为RFE的模型.

4. 扩展高级特征：增加特征
    * 多项式法：x1,x2可扩展为x1^2, x2^2, x1x2
    * 若干特征相加/相减/相乘/相除
    
5. 特征降维：去除特征
    * PCA

'''

# 低方差滤波:去除方差很小的特征
def checkLowVarFeat():
    # sklearn有VarianceThreshold
    variables = train[['Item_Weight', 'Item_Visibility','Item_MRP', 'Outlet_Establishment_Year']]
    featVar = variables.var()
    variables = variables.columns
    variable = []
    for i in range(0, len(featVar)):
        if featVar[i] >= 10:   # 如果方差大于10则保留在variable 里边
            variable.append(variables[i])
            
            
def featShrink(train, label):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    

    # 高相关滤波
    variables = train[['Item_Weight', 'Item_Visibility','Item_MRP', 'Outlet_Establishment_Year']]
    featCorr = variables.corr()  # 如果相关性超过0.5-0.6，则考虑删除一列
    
    # 随机森林评估特征重要性
    from sklearn.ensemble import RandomForestRegressor
    train = train.drop(['Item_Identifier', 'Outlet_Identifier'], axis =1)
    
    train = pd.get_dummies(train)  # 先全部数字化
    
    model = RandomForestRegressor(random_state = 1, max_depth = 10)
    model.fit(train, label)
    
    features = train.columns
    importances = model.feature_importances_
    indices = np.argsort(importances[0:10])     # 前10个重要特征
    plt.title('feature importance')
    plt.barh(range(len(indices)), importances[indices], color = 'b', align = 'center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('relative importance')
    plt.show()
    
    from sklearn.feature_selection import SelectFromModel  # 也可用sklearn的模型选择，根据权重选择特征
    feature = SelectfromModel(model)
    Fit = feature.fit_transform(train, label)
    
# 运行调试区
if __name__=='__main__':
    train, label = loadData()
    newtrain = checkHighPercNullFeat(train)


