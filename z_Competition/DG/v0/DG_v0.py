#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 19:40:18 2018

@author: suliang
"""

# 文件读取的基础知识
# f.read() 读取整个文件，放在一个字符串变量中，不适合读取超大文件，容易内存爆掉
# f.readline() 读取一行内容，放在一个字符串变量中
# f.readlines() 一次读取所有行，但按行返回每行，比如：for line in f.readlines():

# 大文件读取方法1: 分块读取
# 

# 大文件读取方法2: 按行读取
# with open方式读取大文件，参数'rb'方式速度最快，是参数'r'方式的5-6倍速
# with open() as f:
#   for line in f:
#       print(line)


'''
with open('pi_30_digits.txt') as f:    
    for line in f:
        print(line.rstrip())


        
import pandas as pd
f = open('pi_million_digits.txt')
reader = pd.read_csv(f, sep=',', iterator=True)
flag = True
chunkSize = 100
chunks = []
while flag:
    try:
        chunk = reader.get_chunk(chunkSize)  # 读取固定chunksize大小
        chunks.append(chunk)     # chunks为list, 每个list的元素为一个df(100行)
    except StopIteration:
        flag = False
        print("Iteration is stopped.")
df = pd.concat(chunks, ignore_index=True)
print(df)
'''

#-------main------------  
'''
Hardware: 1.3Ghz intel i5, 8G DDR3

>dataimport finish! last time = 32s
>featureing engineering finish! last time = 574s
>fit finish! last time = 490s
>predict finish! last time = 3s
>writing to document finish! last time = 0s
>ALL Finished!!!.......

'''
      
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import datetime

# 导入数据
starttime = datetime.datetime.now()
df_train = pd.read_csv('train_set.csv')
df_train.drop(columns = ['article', 'id'], inplace = True)
df_test = pd.read_csv('test_set.csv')
df_test.drop(columns = ['article'], inplace = True)

lasttime = datetime.datetime.now() - starttime
print('import finish! last time = {}s'.format(lasttime.seconds))

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


# 建模 + 拟合
starttime = datetime.datetime.now()
logclf = LogisticRegression(C=4, dual = True)
logclf.fit(x_train, y_train)

lasttime = datetime.datetime.now() - starttime
print('fit finish! last time = {}s'.format(lasttime.seconds))


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



