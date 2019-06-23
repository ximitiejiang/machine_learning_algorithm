#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:59:16 2019

@author: ubuntu
"""
from collections import namedtuple
import numpy as np

# KdTree 的节点


class KdNode:
    def __init__(self, axis, point, left, right):
        """
        :param axis 决定这个节点以哪一个轴进行分类
        :param point 保存这个节点的值
        :param left 左节点
        :param right 右节点
        """
        self.axis = axis
        self.point = point
        self.left = left
        self.right = right

class KdTree:
    """参考：https://github.com/TensShinet/learn_statistical-learning-method/blob/master/code/%E7%AC%AC3%E7%AB%A0%20k%E8%BF%91%E9%82%BB%E6%B3%95(KNearestNeighbors)/KNN.py"""
    def load_data(self, data):
        # 加载数据
        self.data = data

    def init_args(self):
        """
        通过递归的方法创建 KdTree
        """
        data = self.data
        # k 表示数据的维度
        k = len(data[0])
        # node_num 记录数据集的长度
        self.node_num = len(data)
        # 递归创建节点

        def create_node(axis, data_set):
            # 递归终止条件，数据结构为空，这一层递归返回，也就是说叶节点的再往下的时候，得到节点为 None
            if not data_set:
                return None
            # 排序取中点
            data_set.sort(key=lambda x: x[axis])
            point_pos = len(data_set) // 2     # 此处算出的位置：如果是奇数
            point_media = data_set[point_pos]  # 中位数，如果是奇数个则正好取中间一个，如果是偶数个，则取中间两个的后一个
            # 取余数使 axis 始终在 [0, k] 之间
            next_axis = (axis + 1) % k
            # 递归创建节点
            return KdNode(axis, point_media,
                          create_node(next_axis, data_set[0:point_pos]),
                          create_node(next_axis, data_set[point_pos+1:]))
        # 根节点
        self.root = create_node(0, data)

    # 前序遍历树
    def pre_order(self, ele):
        if not ele:
            return
        self.pre_order(ele.left)
        print(ele.point)
        self.pre_order(ele.right)

# 计算两点之间的距离


def compute_dist(l1, l2):
    # 兼容数组和 np.array
    try:
        return np.linalg.norm(l1 - l2)
    except:
        return np.linalg.norm(np.array(l1) - np.array(l2))


class KNN:
    def __init__(self, kdtree, point):
        """
        :param kdtree 已加载好数据的 kdtree
        :param point 目标点
        """
        self.kdtree = kdtree
        self.point = point

    def add_node(self, point, num=-1):
        """
        :param num 
        如果要求的最近点的数目大于已有最近点的数目，则直接向最近点中加入这个点，此时 num = -1
        如果要求的最近点的数目已经满足已有最近点的数目，则与距离最远的比较，距离比他大就不变，比他小就替换掉
        """
        # 给 tupple 取名字。比如_Result((1, 2)) 输出_Result(dist=1, point=2)
        _Result = namedtuple('_Result', 'dist point')
        # 计算距离
        dist = compute_dist(self.point, point)
        # 组成点和距离的 tupple
        r = _Result(dist, point)
        # 加入这个点
        self.close_nodes.append(r)
        # 排序
        self.close_nodes.sort(key=lambda x: x.dist)

        # 判断这个点的是否进入最近点集合中
        if num == -1:
            return
        else:
            self.close_nodes = self.close_nodes[:num]

    def find_nearest_with_num(self, num=1):
        if num > self.kdtree.node_num:
            print('要找的节点数目，大于树节点的数目')
            return
        self.close_nodes = []
        k = len(self.point)
        target_point = self.point

        def travel(current):
            if not current:
                # 如果当前点是 None
                return
            axis = current.axis
            current_point = current.point
            # 选择更近的一个点
            near_point, far_point = [current.left, current.right] if target_point[
                axis] <= current_point[axis] else [current.right, current.left]

            travel(near_point)
            # 递归遍历回归以后
            if len(self.close_nodes) < num:
                self.add_node(current_point)
            else:
                # 检查当前节点及远边节点满不满足加入的条件
                max_dist = self.close_nodes[num-1].dist
                # 这里算的是垂直轴的距离，如果最远点比这个垂直轴的距离还要小，那么当前点和另一边的点
                # 一定更远
                if max_dist <= abs(current_point[axis] - target_point[axis]):
                    return

                self.add_node(current_point, num)
                travel(far_point)
        # 遍历根节点
        travel(self.kdtree.root)
        return self.close_nodes

class GeneraterData(object):
    def __init__(self):
        pass
    def mul_cla(self, rad_seed=1, cla=2, num=50):
        """
            rad_seed 是 numpy.random.seed 如果 这个值保持一致，生成的随机数一样
            cla 是 有几类的意思，默认返回两类
            start 是随机数开始范围
            end 是随机数字结束范围，但是不包括
            返回都是整数
        """
        np.random.seed(rad_seed)
        data = {}
        for i in range(cla):
            point_x1 = np.random.randint(0 * i, 100 * i, num)
            point_x2 = np.random.randint(0 * i, 100 * i, num)
            if not data['x1']:
                data['flag'] = np.ones((1, num)) * (i + 1) 
                data['x1'] = point_x1
                data['x2'] = point_x2
            else:
                data['x1'] = np.hstack((data['x1'], point_x1))
                data['x2'] = np.hstack((data['x2'], point_x2))
        
        return data
    
    def ser_point(self, rad_seed=1, num=100, max_number=200, min_number=0):
        np.random.seed(rad_seed)
        
        point_x1 = np.random.randint(min_number, max_number, num)
        point_x2 = np.random.randint(min_number, max_number, num)

        data = [[x, y] for x, y in zip(point_x1, point_x2)]

        return data
    
def main():
    # 创建 100 个点
    gd = GeneraterData()
    data = gd.ser_point(num=100)
    # 创建 KdTree
    kd = KdTree()
    kd.load_data(data)
    kd.init_args()
    # 选择一个点
    point = [141, 115]
    # 创建 KNN
    knn = KNN(kd, point)
    # 找最近的 3 个点
    result_points = knn.find_nearest_with_num(num=3)
    # 用
    test_a = []
    for i in range(100):
        dist = compute_dist(point, data[i])
        test_a.append(dist)
    test_a.sort()
    print(result_points)
    print(test_a[0:3])


if __name__ == "__main__":
    main()