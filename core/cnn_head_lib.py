#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:15:42 2019

@author: ubuntu
"""
import torch.nn as nn


class AnchorGenerator():
    
    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
    
    def get_base_anchors(self):
        """获得base anchors，也就是"""
        
    
class SSDHead(nn.Module):
    
    def __init__(self, 
                 input_size=300,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
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
        num_anchors = 
        
        # 创建分类分支，回归分支
        cls_convs = []
        reg_convs = []
        for i in range(len(in_channels)):
            cls_convs.append(
                    nn.Conv2d(in_channels[i], num_anchors[i], kernel_size=3, padding=1))
            reg_convs.append(
                    nn.Conv2d(in_channels[i], num_anchors[i], kernel_size=3, padding=1))
        
        # 计算
        
        # 生成anchor
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        
    
    def init_weight(self):
        for m in self.modules():
            