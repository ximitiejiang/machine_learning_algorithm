#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:44:28 2019

@author: ubuntu
"""
import numpy as np

def bbox_ious(bboxes1, bboxes2):
    """用于计算两组bboxes中每2个bbox之间的iou(包括所有组合，而不只是位置对应的bbox)
    bb1(m, 4), bb2(n, 4), 假定bb1是gt_bbox，则每个gt_bbox需要跟所有anchor计算iou，
    也就是提取每一个gt，因此先从bb1也就是bb1插入轴，(m,1,4),(n,4)->(m,n,4)，也可以先从bb2插入空轴则得到(n,m,4)"""
    # 在numpy环境操作(也可以用pytorch)
    bb1 = bboxes1.numpy()
    bb2 = bboxes2.numpy()
    # 计算重叠区域的左上角，右下角坐标
    xymin = np.max(bb1[:, None, :2] , bb2[:, :2])  # (m,2)(n,2) -> (m,1, 2)(n,2) -> (m,n,2)
    xymax = np.min(bb1[:, 2:] , bb2[:, None, 2:])  # (m,2)(n,2) -> (m,1, 2)(n,2) -> (m,n,2)
    # 计算重叠区域w,h
    wh = xymax - xymin # (m,n,2)-(m,n,2) = (m,n,2)
    # 计算重叠面积和两组bbox面积
    area = wh[:, :, 0] * wh[:, :, 1] # (m,n)
    area1 = (bb1[:, 2] - bb1[:, 0]) * (bb1[:, 3] - bb1[:, 1]) # (m,)*(m,)->(m,)
    area2 = (bb2[:, 2] - bb2[:, 0]) * (bb2[:, 3] - bb2[:, 1]) # (n,)*(n,)->(n,)
    # 计算iou
    ious = area / (area1 + area2[:,None,:] - area)     #(m,n) /[(m,)+(1,n)-(m,n)] -> (m,n) / (m,n)
    
    return ious  # (m,n)


def bbox2delta(prop, gt):
    pass


def delta2bbox():
    pass    


def draw_bbox():
    pass


if __name__ == "__main__":
    draw_bbox()