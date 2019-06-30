#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:30:18 2019

@author: suliang
"""
import numpy as np
import matplotlib.pyplot as plt
from terminaltables import AsciiTable


def generate_table(str_list):
    """用AsciiTable库生成表格，分三步
    1. 创建嵌套list[[]]，内层每个list就是一行
    2. 增加wrapper: AsciiTable(table_list)
    3. 转换成字符table: AsciiTable(table_list).table
    4. 打印
    """    
    # 生成一行table
    table_data = [str_list]  # list嵌套: 内层每个list就是一行
    table = AsciiTable(table_data).table
    return table

def vis_boundary(feats, labels, model, title = None, plot_step=0.02):
    """可视化分隔边界，可适用于线性可分和非线性可分特征，比较普适
    feats
    label_preds(list)
    model: 可视化模型，需含有单点预测函数model.predict_single()
    """
    assert feats.shape[1] == 2, 'feats should be 2 dimention data.'
    assert feats.ndim == 2 # 只能可视化边界二维特征
    xmin, xmax = feats[:,0].min(), feats[:,0].max()
    ymin, ymax = feats[:,1].min(), feats[:,1].max()
    xx, yy = np.meshgrid(np.arange(xmin, xmax, plot_step),
                         np.arange(ymin, ymax, plot_step))
    xx_flatten = xx.flatten()
    yy_flatten = yy.flatten()
    z = []
    for i in range(len(xx_flatten)):
        point = np.array([xx_flatten[i], yy_flatten[i]]) 
        z.append(model.predict_single(point))    # 获得预测
    zz = np.array(z).reshape(xx.shape).astype(np.int8)
    # 绘制等高线颜色填充
    plt.figure()
    plt.subplot(1,1,1)
    plt.contourf(xx, yy, zz, cmap=plt.cm.Paired)
    # 绘制训练数据
    plt.scatter(np.array(feats)[:,0], 
                np.array(feats)[:,1], 
                c = np.array(labels).flatten() * 64 + 128)
    if title:
        model_name = title
    else:
        model_name = 'model'
    plt.title('predict boundary of ' + model_name)
    