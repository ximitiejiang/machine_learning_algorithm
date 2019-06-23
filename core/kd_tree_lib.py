#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:59:16 2019

@author: ubuntu
"""
import numpy as np
from .base_model import BaseModel


points = np.array([[-4.6, -10.55],
                   [-4.96, 12.61],
                   [1.75, 12.26],
                   [15.31,-13.16],
                   [7.83, 15.70],
                   [14.63, -0.35],
                   [-6.88, -5.4],
                   [-2.96, -0.5],
                   [7.75, -22.68],
                   [10.8, -5.03],
                   [1.24, -2.86],
                   [17.05, -12.79],
                   [6.27, 5.5]])

class KdTree(BaseModel):
    
    def __init__(self, feats, labels):
        """ kd tree算法: kd是指k dimension tree，也就是把k维特征数据存储到树结构中
        原理介绍参考：https://www.joinquant.com/view/community/detail/c2c41c79657cebf8cd871b44ce4f5d97
        https://www.cnblogs.com/lysuns/articles/4710712.html
                     
        代码参考：https://github.com/tsoding/kdtree-in-python/blob/master/main.py, 该方法把树放置到一个字典中
        https://blog.csdn.net/u010551621/article/details/44813299#comments, 该方法是采用node来存放数据
        Args:
            k: 表示要选择的k个近邻
        """
        self.k = 2  # k为
        self.tree = self.build_kdtree(feats)



class KD_node:
    def __init__(self, point=None, split=None, LL = None, RR = None):
        """
        point:数据点
        split:划分域
        LL, RR:节点的左儿子跟右儿子
        """
        self.point = point
        self.split = split
        self.left = LL
        self.right = RR
    
def createKDTree(root, data_list):
    """
    root:当前树的根节点
    data_list:数据点的集合(无序)
    return:构造的KDTree的树根
    """
    LEN = len(data_list)
    if LEN == 0:
        return
    #数据点的维度
    dimension = len(data_list[0])
    #方差
    max_var = 0
    #最后选择的划分域
    split = 0;
    for i in range(dimension):
        ll = []
        for t in data_list:
            ll.append(t[i])
        var = computeVariance(ll)
        if var > max_var:
            max_var = var
            split = i
    #根据划分域的数据对数据点进行排序
    data_list.sort(key=lambda x: x[split])
    #选择下标为len / 2的点作为分割点
    point = data_list[LEN / 2]
    root = KD_node(point, split)
    root.left = createKDTree(root.left, data_list[0:(LEN / 2)])
    root.right = createKDTree(root.right, data_list[(LEN / 2 + 1):LEN])
    return root
 
 
def computeVariance(arrayList):
    """
    arrayList:存放的数据点
    return:返回数据点的方差
    """
    for ele in arrayList:
        ele = float(ele)
    LEN = len(arrayList)
    array = numpy.array(arrayList)
    sum1 = array.sum()
    array2 = array * array
    sum2 = array2.sum()
    mean = sum1 / LEN
    #D[X] = E[x^2] - (E[x])^2
    variance = sum2 / LEN - mean**2
    return variance

def findNN(root, query):
    """
    root:KDTree的树根
    query:查询点
    return:返回距离data最近的点NN，同时返回最短距离min_dist
    """
    #初始化为root的节点
    NN = root.point
    min_dist = computeDist(query, NN)
    nodeList = []
    temp_root = root
    ##二分查找建立路径
    while temp_root:
        nodeList.append(temp_root)
        dd = computeDist(query, temp_root.point)
        if min_dist > dd:
            NN = temp_root.point
            min_dist = dd
        #当前节点的划分域
        ss = temp_root.split
        if query[ss] <= temp_root.point[ss]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right
    ##回溯查找
    while nodeList:
        #使用list模拟栈，后进先出
        back_point = nodeList.pop()
        ss = back_point.split
        print "back.point = ", back_point.point
        ##判断是否需要进入父亲节点的子空间进行搜索
        if abs(query[ss] - back_point.point[ss]) < min_dist:
            if query[ss] <= back_point.point[ss]:
                temp_root = back_point.right
            else:
                temp_root = back_point.left
 
            if temp_root:
                nodeList.append(temp_root)
                curDist = computeDist(query, temp_root.point)
                if min_dist > curDist:
                    min_dist = curDist
                    NN = temp_root.point
    return NN, min_dist
 
 
def computeDist(pt1, pt2):
    """
    计算两个数据点的距离
    return:pt1和pt2之间的距离
    """
    sum = 0.0
    for i in range(len(pt1)):
        sum = sum + (pt1[i] - pt2[i]) * (pt1[i] - pt2[i])
    return math.sqrt(sum)


if __name__ == "__main__":
    # 1. 创建kd tree存放特征数据
    root = creatKDTree()