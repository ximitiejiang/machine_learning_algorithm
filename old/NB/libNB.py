#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 08:32:30 2018

@author: suliang
"""

def loadDataSet(): # 创建测试样本
    postingList = [['my', 'dog', 'has', 'flea', 'problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cut','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec
    
    
def createVocabList(data):
    vocabSet = set([])   # 新建一个集合（集合为python另一种数据结构）
    for document in data:
        vocabSet = vocabSet | set(document)  # 先对样本去除重复值set，然后集合并集
    return list(vocabSet)  # 反馈词汇表list格式
        

# 词集模型的创建：词集模型中每个词只能出现一次
def setOfWord2Vec(vocabList, inputSet):  # 基于vocabList，把新句子的所有单词转化为向量
    # word2vec的逻辑是：用一个词汇表那么长的0/1向量来表示一句话
    # 一句话中的每一个单词在词汇表位置处用1表示，其他没出现的单词位置都保持0
    # 从而每一句话都有了独一无二的一个二进制编码
    returnVec = [0*i for i in range(len(vocabList))] # 创建一个有词汇表那么长的全0向量
    for word in inputSet:  # 取出一句话中每一个单词
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1   # 如果该单词存在词汇表，词汇标
        else: 
            print('the word: {} is not in my vocabulary'.format(word))
    return returnVec

# 词袋模型的创建：词可以出现多次
def bagOfWord2Vec(vocabList, inputSet):
    returnVec = [0*i for i in range(len(vocabList))] # 创建一个有词汇表那么长的全0向量
    for word in inputSet:  # 取出一句话中每一个单词
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1   # 如果该单词存在词汇表，词汇标
        else: 
            print('the word: {} is not in my vocabulary'.format(word))
    return returnVec    
    

def trainNB0(trainMatrix, category):
    import numpy as np
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(category)/float(numTrainDocs)
    
    p0Num = np.zeros(numWords)  # 初始化概率
    p1Num = np.zeros(numWords)
    p0Denom = 0
    p1Denom = 0
    
    for i in range(numTrainDocs):
        if category[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        
    p1Vec = p1Num/p1Denom
    p0Vec = p0Num/p0Denom
    
    return p0Vec, p1Vec, pAbusive            
    

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    pass


def testingNB():
    pass

#------------main-----------------
    
datalist, vec = loadDataSet()
vocablist = createVocabList(datalist)

trainMat = []
for i in datalist:
    trainMat.append(setOfWord2Vec(datalist, vec))
    
p0V, p1V, pAb = trainNB0(trainMat, vec)

newSentence = 'this book is the best book on python or on M.L. I have ever laid eyes upon'


