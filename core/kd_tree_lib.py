#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:59:16 2019

@author: ubuntu
"""
from .base_model import BaseModel
from collections import namedtuple
import numpy as np


class KdNode:
    def __init__(self, axis, point, label, left, right):
        """ 存放每一个节点的对象，每个节点存放唯一一个样本
        采用对象存放数据的优点是调用方便，obj.point可直接调用，obj.left.left.right.point可嵌套调用
        Args:
            axis 决定这个节点以哪一个轴进行分类
            point 保存这个节点的值
            left 左节点
            right 右节点
        """
        self.axis = axis
        self.point = point
        self.label = label
        self.left = left
        self.right = right


class KdTree(BaseModel):
    def __init__(self, feats, labels, k=5, norm=False):
        """ kdtree algorithm lib
        特点：本质上是采用kdtree作为特征预存储的knn算法，支持二分类和多分类，无模型参数，支持线性可分和非线性可分数据
        相对于knn的改进：由于特征存放在kdtree这种二叉树中，在做预测时会减少计算量
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)
            k: k个近邻
        """
        super().__init__(feats, labels, norm=norm)
        self._k = k
        self.n_nodes = feats.shape[0]
        assert k <= self.n_nodes, 'num of k should be less than num of samples.'
        
        self.root = self.create_kdtree(0, self.feats, self.labels) # 先创建kdtree存放特征/标签
        self.model_dict['model_name'] = 'KdTree' + '_k' + str(self._k)
        self.model_dict['k'] = self._k
        self.model_dict['kdtree'] = self.root
        self.trained = True # 由于kdtree没有训练过程，只要root创建后，就把trained设置为True，从而可保存
    
    def create_kdtree(self, axis=0, feats=None, labels=None):
        """递归创建kdtree存放数据，生成的kdtree如下结构, 最终会到达None
                       (7,2)
                    /         \
               (5,4)           (9,6)
              /     \          /   \
           (2,3)   (4,7)     (8,1)  None
           /   \    /   \     /   \
         None None None None None None
        axis: 指定进行二叉树分割的初始坐标轴
        feats: 特征
        labels: 标签
        """
        n_feats = feats.shape[1]
        if len(feats) == 0:    # 如果数据为空，跳出递归
            return None
        labels = labels[np.argsort(feats[:, axis])] # labels先按照feats的排序进行更新
        feats = feats[np.argsort(feats[:, axis])]   # 基于当前axis进行样本排序
        
        median_idx = len(feats) // 2                 # 中位数index(偶数则为两个数中的第二个)
        point_median = feats[median_idx]              # 中位数值
        label_median = labels[median_idx]             # 中位数标签
    
        next_axis = (axis + 1) % n_feats
        
        return KdNode(axis, 
                      point_median,
                      label_median,
                      self.create_kdtree(next_axis, feats[0:median_idx, :], labels[0:median_idx]),
                      self.create_kdtree(next_axis, feats[median_idx+1:, :], labels[median_idx+1:]))
    
    def finde_nearest_neighbour(self, current_node, target_point):
        """递归寻找最邻近点
        Args:
            current_node: 当前子树
            target_point: 目标样本
        """
        def travel(current_node, target_point, nearest_nodes):
            """递归寻找最邻近点"""
            if not current_node: # 当前node为None
                return
            axis = current_node.axis
            current_point = current_node.point
            current_label = current_node.label
            
            # 递归下行，直到找到叶子结点
            near_point, far_point = [current_node.left, current_node.right] \
                                    if target_point[axis] <= current_point[axis] \
                                    else [current_node.right, current_node.left]
            travel(near_point, target_point, nearest_nodes)  # 递归直到最后为None的节点才执行下面的代码
            
            # 下行递归结束后，改为回退
            if len(nearest_nodes) < self._k:    #如果已保存的近邻点数量少于需要的k个，则添加当前点
                self.add_node(current_point, current_label, target_point, nearest_nodes)
            else:
                max_dist = nearest_nodes[self._k -1].dist  # 如果已保存足够近邻点个数，则从已保存的近邻点提取最大距离(已排序的最后一个点)
                if max_dist <= abs(current_point[axis] - target_point[axis]): # 如果最大距离比当前点小，则当前点没用(该距离为在某一坐标轴方向的距离)
                    return
                else:
                    self.add_node(current_point, current_label, target_point, nearest_nodes)  # 如果最大距离点比当前点大，则添加当前点
                    travel(far_point, target_point, nearest_nodes)   # 只要添加一个点，就要检查该点对应的另外一个远分支是否有更近的点    
        
        nearest_nodes = []  # 存放近邻点
        travel(self.root, target_point, nearest_nodes)
        return nearest_nodes[:self._k]   # 最终得到的nearest_nodes个数有可能多于k个，则截取前k个距离最小的即可
    
    def add_node(self, point, label, target_point, nearest_nodes):
        """处理要添加的点：计算距离，组合坐标与距离，加入新点，排序
        Args:
            point: 待添加到近邻点列表的某点
            target_point: 目标样本
        如果要求的最近点的数目大于已有最近点的数目，则直接向最近点中加入这个点，此时 num = -1
        如果要求的最近点的数目已经满足已有最近点的数目，则与距离最远的比较，距离比他大就不变，比他小就替换掉
        """

        result = namedtuple('result', 'dist point label') # 创建一个namedtuple(name, var_list), 地一个参数是该tuple的变量名，第二个参数是存放的变量名(空格隔开)
        dist = self.compute_dist(target_point, point)
        r = result(dist, point, label)
        nearest_nodes.append(r)
        nearest_nodes.sort(key=lambda x: x.dist) 

    
    @staticmethod
    def compute_dist(l1, l2):
        # 兼容数组和 np.array
        try:
            return np.linalg.norm(l1 - l2)   # 默认求l2范数
        except:
            return np.linalg.norm(np.array(l1) - np.array(l2))    
        
    def predict_single(self, sample_single):
        """单样本预测
        Args:
            sample_simple: (m,) 一个单样本
        """
        assert isinstance(sample_single, np.ndarray), 'data should be ndarray.'
        assert (sample_single.shape[0]==1 or sample_single.ndim==1), 'data should be flatten data like(m,) or (1,m).'
        assert (self._k % 2 == 1), 'k should be odd number.'
        
        nearest_points = self.finde_nearest_neighbour(self.root, sample_single)
        
        count_dict = {}  # 存储  {标签：个数}
        for p in nearest_points:
            label = p.label
            count_dict[label] = count_dict.get(label, 0) + 1

        # get most counted label
        label, _ = max(zip(count_dict.keys(), count_dict.values()))   # 字典排序
        return label