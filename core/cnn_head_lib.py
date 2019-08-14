#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:15:42 2019

@author: ubuntu
"""
import torch.nn as nn
from module_factory import registry

class AnchorGenerator():
    
    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
    
    def get_base_anchors(self):
        """获得base anchors，也就是"""
        
def cal_ssd_anchor_params(input_size):
    """生成ssd的anchor所需参数"""    
    if input_size==300: # voc 300
        min_sizes = [30, 60, 111, 162, 213, 264]
        max_sizes = [60, 111, 162, 213, 264, 315]
    
    base_sizes = min_sizes
    strides = anchor_strides
    ctr = ((stride - 1) / 2., (stride - 1) / 2.)
    scales = 
    ratios = 
    
    return 


@registry.register_module    
class SSDHead(nn.Module):
    """分类回归头：
    1. 分类回归头的工呢过：保持尺寸w,h不变，变换层数进行层数匹配，把特征金字塔的每一级输出层变换成num_anchor
    其中分类层：输出层数 = 目标分类指标数量 = 类别数*每个特征像素映射到原图后上面放置的anchor个数
        这里anchor个数在特征金字塔不同层不同，分别是(4,6,6,6,4,4)
    其中回归层：输出层数 = 目标回归指标数量 = 回归坐标数*每个特征像素映射到原图后上面放置的anchor个数
           -----------------
          /                 \
        [cls]               [reg]
        3x3(512 to 81*4)    3x3(512 to 4*4)
        3x3(1024 to 81*6)   3x3(1024 to 4*6)
        3x3(512 to 81*6)    3x3(512 to 4*6)
        3x3(256 to 81*6)    3x3(256 to 4*6)
        3x3(256 to 81*4)    3x3(256 to 4*4) 
        3x3(256 to 81*4)    3x3(256 to 4*4) 
    
    2. anchor生成机制：
    其中anchor的个数(4,6,6,6,4,4)是根据经验
    
    
    """
    
    def __init__(self, 
                 input_size=300,
                 num_classes=21,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 num_anchors=(4, 6, 6, 6, 4, 4),
                 anchor_strides=(8, 16, 32, 64, 100, 300),
                 

                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 **kwargs): # 增加一个多余变量，避免修改cfg, 里边有一个type变量没有用
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.cls_out_channels = num_classes

        # 创建分类分支，回归分支
        cls_convs = []
        reg_convs = []
        for i in range(len(in_channels)):
            cls_convs.append(
                    nn.Conv2d(in_channels[i], num_anchors[i], kernel_size=3, padding=1))
            reg_convs.append(
                    nn.Conv2d(in_channels[i], num_anchors[i], kernel_size=3, padding=1))
        self.cls_convs = nn.ModuleList(cls_convs)
        self.reg_convs = nn.ModuleList(reg_convs)
        
        # 生成anchor所需标准参数
        min_sizes = [30, 60, 111, 162, 213, 264]
        max_sizes = [60, 111, 162, 213, 264, 315]
        
        base_sizes = min_sizes
        scales = (1, 1,414, )
        ratios = [1, 1/2, 2]
        ctr = ((stride - 1) / 2., (stride - 1) / 2.)
        
        base_sizes, scales, ratios = cal_ssd_anchor_params()
        
        # 生成anchor
        self.anchor_generators = []
        for base_size, scale, ratio in zip(base_sizes, scales, ratios):
            self.anchor_generators.append(AnchorGenerator(base_size, scale, ratio, scale_major=False, ctr=ctr))
        
    
    def init_weight(self):
        for m in self.modules():
            pass
        
    def forward(self, x):
        pass
    
    def loss(self):
        pass
    
    def loss_single(self):
        pass
    
    def get_bboxes(self):
            