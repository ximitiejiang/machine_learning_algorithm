#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:15:42 2019

@author: ubuntu
"""
import numpy as np
import torch
import torch.nn as nn
from utils.module_factory import registry
from utils.weight_init import xavier_init


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
        self.cls_convs = nn.ModuleList(cls_convs) # 由于6个convs是并行分别处理每一个特征层，所以不需要用sequential
        self.reg_convs = nn.ModuleList(reg_convs)
        
        # 生成anchor所需标准参数
        # base sizes：表示一组anchor的基础大小，然后乘以scale(比例)，变换宽高比(ratio)
        min_sizes = [30, 60, 111, 162, 213, 264]
        max_sizes = [60, 111, 162, 213, 264, 315]
        base_sizes = min_sizes
        # strides: 表示两组anchor中心点的距离
        strides = anchor_strides
        # centers: 表示每组anchor的中心点坐标
        centers = []
        for stride in strides:
            centers.append(((stride - 1) / 2., (stride - 1) / 2.))
        # scales: 表示anchor基础尺寸的放大倍数
        scales = []       
        for max_size, min_size in zip(max_sizes, min_sizes):
            scales.append([1., np.sqrt(max_size / min_size)])  # ssd定义2种scale(1,sqrt(k))
        # ratios：表示anchor的高与宽的比例
        ratios = ([1, 1/2, 2], [1, 1/2, 2, 1/3, 3], [1, 1/2, 2, 1/3, 3], [1, 1/2, 2, 1/3, 3], [1, 1/2, 2], [1, 1/2, 2])
        # 生成anchor
        self.anchor_generators = []
        for base_size, scale, ratio, ctr in zip(base_sizes, scales, ratios, centers):
            anchor_generator = AnchorGenerator(base_size, scale, ratio, scale_major=False, ctr=ctr)
            # 截取一定个数的anchors作为base anchor
            keep_anchor_indics = range(0, len(ratio)+1)   # 保留的anchor: 2*3的前(0-3), 2*5的前(0-5)
            anchor_generator.base_anchors = anchor_generator.base_anchors[keep_anchor_indics]
        
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform", bias=0)
        
    def forward(self, x):
        cls_scores = []
        bbox_preds = []
        for i, feat in enumerate(x):
            cls_scores.append(self.cls_convs[i](feat))
            bbox_preds.append(self.reg_convs[i](feat))
        return cls_scores, bbox_preds 
    
    def get_losses(self, cls_scores, bbox_preds, 
                   gt_bboxes, gt_labels, 
                   img_metas, cfg):
        """在训练时基于前向计算结果，计算损失"""
        
        # 基于base anchors生成
        multi_layers_anchors = []
        for i in range(len(img_metas)):
            anchors = self.anchor_generators.grid_anchors()
            multi_layers_anchors.append(anchors)
            
    
    def loss_single(self):
        pass
    
    def get_bboxes(self):
        """在测试时基于前向计算结果，计算bbox预测值，此时前向计算后不需要算loss，直接算bbox"""
        pass


class AnchorGenerator():
    """生成base anchors和grid anchors"""
    def __init__(self, base_size, scales, ratios, scale_major=False, ctr=None):
        self.base_size = base_size
        self.scales = np.array(scales)
        self.ratios = np.array(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        
        self.base_anchors = self.get_base_anchors()
    
    def get_base_anchors(self): 
        """生成单个特征图的base anchors"""
        w, h = self.base_size, self.base_size
        # 准备中心点坐标
        if self.ctr is None:
            x_ctr, y_ctr = 0.5 * (w - 1), 0.5 * (h - 1) 
        else:
            x_ctr, y_ctr = self.ctr
        # 准备宽高的比例: ratio=h/w=2则h=sqrt(2)*h0, w=1/sqrt(2) * w0
        h_ratios = np.sqrt(self.ratios) #(n,)
        w_ratios = 1 / h_ratios         #(n,)
        # 计算变换后的w', h'
        if self.scale_major:
            w_new = (w * w_ratios[:, None] *self.scales[None, :]).reshape(-1)    # (n,1)*(1,m)->(n,m)->(n*m,)
            h_new = (h * h_ratios[:, None] *self.scales[None, :]).reshape(-1)
        else:
            w_new = (w * self.scales[:, None] * w_ratios[None, :]).reshape(-1)    # (m,1)*(1,n)->(m,n)->(m*n,)
            h_new = (h * self.scales[:, None] * h_ratios[None, :]).reshape(-1)
        # 计算坐标xmin,ymin,xmax,ymax
        base_anchors = np.stack([x_ctr - 0.5 * (w_new - 1), 
                                 y_ctr - 0.5 * (h_new - 1),
                                 x_ctr + 0.5 * (w_new - 1), 
                                 y_ctr + 0.5 * (h_new - 1)], axis=-1).round()  # (m*n, 4))
        
        return torch.tensor(base_anchors)
    
    def grid_anchors(self, featmap_size, stride):
        """生成单个特征图的网格anchors"""
        #TODO: 检查是否要送入device
        base_anchors = self.base_anchors #(k, 4)
        # 生成原图上的网格坐标
        h, w = featmap_size
        x = np.arange(0, w) * stride  # (m,)
        y = np.arange(0, h) * stride  # (n,)
        xx = np.tile(x, (len(y),1)).reshape(-1)                # (m*n,)
        yy = np.tile(y.reshape(-1,1), (1, len(x))).reshape(-1) # (m*n,)
        # 基于网格坐标生成(xmin, ymin, xmax, ymax)的坐标平移矩阵
        shifts = np.stack([xx, yy, xx, yy], axis=-1)  # (m*n, 4)
        shifts = torch.tensor(shifts)
        # 平移anchors: 相当于该组anchors跟每个平移坐标进行相加，
        # 也就相当于要取每一行坐标跟一组anchor运算，所以用坐标插入空轴而不是用anchor插入空轴
        all_anchors = base_anchors + shifts[:, None, :]   #(b,4)+(k,1,4)->(k, b, 4)
        all_anchors = all_anchors.reshape(-1, 4)  # (k*b, 4)
        return all_anchors
    

if __name__ == "__main__":
    """base_anchor的标准数据
    [[-11., -11.,  18.,  18.],[-17., -17.,  24.,  24.],[-17.,  -7.,  24.,  14.],[ -7., -17.,  14.,  24.]]
    [[-22., -22.,  37.,  37.],[-33., -33.,  48.,  48.],[-34., -13.,  49.,  28.],[-13., -34.,  28.,  49.],[-44.,  -9.,  59.,  24.],[ -9., -44.,  24.,  59.]]
    [[-40., -40.,  70.,  70.],[-51., -51.,  82.,  82.],[-62., -23.,  93.,  54.],[-23., -62.,  54.,  93.],[-80., -16., 111.,  47.],[-16., -80.,  47., 111.]]
    [[ -49.,  -49.,  112.,  112.],[ -61.,  -61.,  124.,  124.],[ -83.,  -25.,  146.,   88.],[ -25.,  -83.,   88.,  146.],[-108.,  -15.,  171.,   78.],[ -15., -108.,   78.,  171.]]
    [[ -56.,  -56.,  156.,  156.],[ -69.,  -69.,  168.,  168.],[-101.,  -25.,  200.,  124.],[ -25., -101.,  124.,  200.]]
    [[ 18.,  18., 281., 281.],[  6.,   6., 293., 293.],[-37.,  57., 336., 242.],[ 57., -37., 242., 336.]]
    """
    
    import sys, os
    path = os.path.abspath("../utils")
    if not path in sys.path:
        sys.path.insert(0, path)
    
    head = SSDHead()
    
    """广播机制的应用: 前提是两个变量从右往左，右边的对应轴size要相同，或者其中一个变量size=0或1
        对其中一个变量插入一个轴，就相当于对他提取每一行，并广播成另一个变量的形状"""
    a = np.ones((3,4))
    b = np.ones((10,4))
    result = a + b[:, None, :]
    
    """广播机制的应用2: 有3个角色分别有各自的攻击力和防御力，各自攻击100个目标分别获取攻击防御力加成"""
    roles = np.array([[1,2],[3,2],[4,1]])           # (3,2)
    objects = np.random.randint(1,10, size=(100,2)) # (100,2)
    result = roles + objects[:, None, :]    # (3,2)+(100,1,2)->(100,3,2)
    
    
        