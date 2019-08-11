#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:54:44 2019

@author: ubuntu
"""

import torch.nn as nn

# %% 最简版ssd vgg16

class SSDVGG16(nn.Module):
    """ 在基础版VGG16结构上ssd修改部分包括：去掉最后一层maxpool然后增加一层maxpool，增加extra convs, l2norm
        img                 (3,  h, w)
        2x    3x3           (64,     )
              3x3           (64,     )
              maxpool       (64, h/2, w/2)
        2x    3x3           (128,        )
              3x3
              maxpool       (128, h/4, w/4)
        3x    3x3           (256          )
              3x3
              3x3
              maxpool       (256, h/8, w/8)
        3x    3x3           (512          )
              3x3
              3x3
              maxpool       (512, h/16, w/16)
        3x    3x3           (512            )
              3x3
              3x3
              maxpool       (512, h/32, w/32)
    """
    arch_setting = {16: [2,2,3,3,3]}  # 16表示vgg16，后边list表示有5个blocks，每个blocks的卷积层数
    
    def __init__(self, 
                 num_classes=2, 
                 in_channels=3, 
                 out_channels=(128, 256, 512, 512, 512)):
        super().__init__()
        self.blocks = self.arch_setting[16]
        
        #构建所有vgg基础层
        vgg_layers = []
        in_channels = 3
        for i, convs in self.blocks:
            out_channels = 64 * (i+1) if i <=3 else 512
            block_layers = self.make_vgg_block(convs, in_channels, out_channels)
            vgg_layers.extend(block_layers)
        vgg_layers.pop(-1) # 去掉最后一层max pool
        self.features = nn.Sequential(*vgg_layers) 
        
        # ssd额外添加maxpool + 2层conv
        self.features.add_module(
                str(len(self.features)), nn.MaxPool2d(kernel_size=3, stride=1, padding=1))# 最后一个maxpool的stride改为1
        self.features.add_module(
                str(len(self.features)), nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)) # 空洞卷积
        self.features.add_module(
                str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
                str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
                str(len(self.features)), nn.ReLU(inplace=True))
        
        # 构建ssd额外卷积层和l2norm层(ssd论文提及)
        self.extra = self.make_extra_block(in_channels=1024)
        self.l2_norm = L2Norm()
        
        # 构建分类模块
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes))
        
        
    def make_vgg_block(self, num_convs, in_channels, out_channels, with_bn=False, ceil_mode=False):
        """构造一个conv + bn + relu + Maxpool的子模块，存成list，最后统一解包放入nn.Sequential()
        """
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if with_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
        return layers
     
    def make_extra_block(self, in_channels):
        """额外增加10个conv，用来获得额外的更多尺度输出
        extra_setting = {300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256)}
        """
        layers = []
        layers.append(nn.Conv2d(in_channels, 256, kernerl_size=1, stride=1, padding=0))
        layers.append(nn.Conv2d(256, 256, kernerl_size=3, stride=2, padding=1)) # s=2
        
        layers.append(nn.Conv2d(256, 512, kernerl_size=1, stride=1, padding=0))
        layers.append(nn.Conv2d(512, 128, kernerl_size=3, stride=1, padding=0))
        
        layers.append(nn.Conv2d(128, 128, kernerl_size=1, stride=2, padding=1))  # s=2
        layers.append(nn.Conv2d(128, 256, kernerl_size=3, stride=1, padding=0))
        
        layers.append(nn.Conv2d(256, 128, kernerl_size=1, stride=1, padding=0))
        layers.append(nn.Conv2d(128, 256, kernerl_size=3, stride=1, padding=0))
        layers.append(nn.Conv2d(256, 128, kernerl_size=1, stride=1, padding=0))
        layers.append(nn.Conv2d(128, 256, kernerl_size=3, stride=1, padding=0))
        return nn.Sequential(*layers)
    
    def init_weight(self):
        pass
    
    def forward(self, x):
        pass
        
class L2Norm(nn.Module):
    def __init__(self):
        pass        
    
    def forward(self, x):
        pass
        