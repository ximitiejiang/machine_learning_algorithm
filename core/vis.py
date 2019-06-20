#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:19:47 2019

@author: ubuntu
"""
import numpy as np

def vis_boundary(model, X, plot_step=0.02):
    """基于模型的预测函数，绘制预测边界
    Args:
        model: 模型对象，必须包含predict函数用于单样本预测
        X(array): (n, 2)训练数据，n个样本，2列特征
        plot_step: 在样本空间meshgrid划分网格的间隔大小
    """
    x_min, x_max = X[:, 0].min()
    y_min, y_max = X[:, 1].min()
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # 把xx, yy都展平，然后
    z = z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, z, cmap=plt.cm.Paired)
    plt.axis('tight')
    
    