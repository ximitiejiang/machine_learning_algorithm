#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:30:18 2019

@author: suliang
"""
import numpy as np
import matplotlib.pyplot as plt
from terminaltables import AsciiTable

def basic_colors():
    colors = [[]]

def city_colors(n_classes, norm=False):
    cityspallete = [
                    128, 64, 128,
                    244, 35, 232,
                    70, 70, 70,
                    102, 102, 156,
                    190, 153, 153,
                    153, 153, 153,
                    250, 170, 30,
                    220, 220, 0,
                    107, 142, 35,
                    152, 251, 152,
                    0, 130, 180,
                    220, 20, 60,
                    255, 0, 0,
                    0, 0, 142,
                    0, 0, 70,
                    0, 60, 100,
                    0, 80, 100,
                    0, 0, 230,
                    119, 11, 32]
    colors = []
    for i in range(int(len(cityspallete)/3)):
        colors.append([cityspallete[i*3], cityspallete[i*3+1], cityspallete[i*3+2]])
    colors = np.array(colors)
    colors = colors[:n_classes]
    if norm:
        colors = colors / 255
    return colors
    

def voc_colors(n_classes, norm=False):
    """生成一组颜色，基于voc语义分割数据集的定义
    返回的是一组(n_class, 3)的rbg值，取值范围(0-255)。
    进一步颜色显示可能需要：1.颜色归一化到(0-1), 
    2.扩展成(n_labels,3): colors = colors[label_list]
    
    Returns:
        colors(array): (n_cls, 3)
    """
    n = n_classes
    colors = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        colors[j * 3 + 0] = 0
        colors[j * 3 + 1] = 0
        colors[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            colors[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            colors[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            colors[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    if norm:
        colors = np.array(colors).reshape(-1,3) / 255
    else:
        colors = np.array(colors).reshape(-1,3)
    return colors


def generate_table(str_list):
    """用AsciiTable库生成表格，分三步
    1. 创建嵌套list[[]]，内层每个list就是一行
    2. 增加wrapper: AsciiTable(table_list)
    3. 转换成字符table: AsciiTable(table_list).table
    4. 打印
    """
    if isinstance(str_list, list) and not isinstance(str_list[0], list):    
        table = AsciiTable([str_list]).table
    elif isinstance(str_list, list) and isinstance(str_list[0], list):  
        table = AsciiTable(str_list).table
    print(table)


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


if __name__ == "__main__":
    colors = city_colors(10, norm=True)
    
    generate_table([['姓名','eason','winnie'],[1,2,3]])