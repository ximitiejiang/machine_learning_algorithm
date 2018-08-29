#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:21:05 2018

@author: suliang
"""

import numpy as np

def loadDataSet(filename):
    df = pd.read_table(filename,header = None)
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values     # 假定所有数据最后一列都是label
    return x, y


def binSplitDataSet(data, feature, value):  #
    mat0 = data[np.nonzero(data[:,feature] > value)[0], :][0]
    mat1 = data[np.nonzero(data[:,feature] <= value)[0], :][0]
    return mat0, mat1

'''
def createTree(data, leafType = regLeaf, errType = regErr, ops = (1,4)):
    feat, val = chooseBestSplit(data, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    rerTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(data, feat, val)
    
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
'''

testMat = np.mat(np.eye(4))

mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
