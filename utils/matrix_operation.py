#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 23:28:38 2019

@author: ubuntu
"""
import numpy as np

def img2col(img, filter_h, filter_w, stride=1, pad=0):
    """用于把图片matrix转换为列数据形式，便于跟权值W进行矩阵点积，本质就是把卷积计算转换为矩阵点积，避免卷积的复杂for循环。
    参考：《Deep Learning from Scratch》, 7.4, 斋藤康毅，日本。
    注意：相对源码做了修改，让输出变为(单滤波器元素，滤波器个数)，也就是每列是一个滤波操作，跟img2col的含义更匹配。
    Args:
        img : (b,c,h,w)4维数组构成的输入数据
        filter_h : 卷积核h
        filter_w : 卷积核w
        stride : 步幅
        pad : 填充
    Returns
        col : 2维数组
    """
    N, C, H, W = img.shape
    out_h = (H + 2*pad - filter_h)//stride + 1  # 计算输出图像实际尺寸
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(img, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')  # 填充2,3轴的行列
    # 核心：初始化 (b, c, h, w, hh, ww), 相当于把c*hh**ww的卷积核进行一次卷积，每张图是h*w次，共计b张图，所以col就是按照卷积次数设置的尺寸。
    # 但这里没有考虑输出层数带来的次数，
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) 

    for y in range(filter_h):
        y_max = y + stride*out_h  
        for x in range(filter_w):
            x_max = x + stride*out_w  
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]  # 取每组滤波器对应像素

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1).T
    return col  # (每个滤波器元素个数chw, 输出滤波器个数o_h*o_w) 也就是每列就是一组滤波器数据，比如(9,16384)


def col2img(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """用于把列数据转换为图片matrix形式，为img2col的逆运算
    Args:
        col :
        input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
        filter_h :
        filter_w
        stride
        pad
    Returns
        img
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


if __name__ == "__main__":
    img = np.random.randint(0, 255, size= (1, 1, 7, 7))
    
    img_col = img2col(img, 5, 5, 1, 0)  # 输入7*7, 滤波器5*5, s=1, p=0, 输出3*3
    
    # 测试一组真实数据
    image = np.random.randint(0, 255, size=(256, 1, 8, 8))
    image_col = img2col(image, 3, 3, 1, 1)  # 输入8*8, 滤波器3*3, s=1, p=1, 输出8*8, 基于这个参数, col shape=(16384,9)
    