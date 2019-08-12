#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% 创建pytorch版本的module

"""继承自nn.Module"""
import torch.nn as nn
class VGG(nn.Module):
    def __init__(self):
        super().__init__()

"""创建模型容器用nn.Sequential(*list), nn.ModuleList(list), nn.ModuleDict(dict)
    - 如果用nn.Sequential(),则不用自己写内部的forwrd，默认采用串联的层计算结果。

    - 如果用nn.ModuleList(),则需要自己写forward来自定义内部层的计算过程。

    - 如果用nn.ModuleDict(),则需要自己写forward来自定义内部层的计算过程。
"""
layers = [nn.Conv2d(3, 64, 3), nn.ReLU()]  #
model = nn.Sequential(*layers)       

layers = [nn.Conv2d(3, 64, 3), nn.ReLU()]
model = nn.ModuleList(layers)

layers = dict(conv1 = nn.Conv2d(), relu1 = nn.ReLU())
model = nn.ModuleDict(layers)

"""创建每一层，可选择的层包括：
    - nn.Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True)
        上面都是默认值，其中stride为步长，padding为边沿填充，dilation为空洞卷积率，bias为偏置
        其中dilation=n就意味着1x1的核像素变成n*n, 所以dilation=1表示没有空洞。
        其中输出宽高计算公式w = (w - k_size + 2*p) / s + 1
    
    - nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False)
        其中ceil_mode表示取整方式，默认是floor下取整

    - nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False)
        其中ceil_mode表示取整方式，默认是floor下取整
    
    - nn.Linear(in_features, out_features, bias=True)
        其中in_features, out_features表示输入输出的特征长度
        注意，Linear层的输入输出只能是batch size个一维数组(b, n), b表示batch_size, n表示特征长度
    
    - nn.ReLU(inplace=False)
        其中inplace是指直接对输入数据进行修改，而不是先生成一个副本再修改
    
    - nn.BatchNorm2d(num_features, eps=12-05, momentum=0.1, affine=True)
        其中num_features为输入特征数，
        其中affine为是否有可学习参数，默认为True，也可设置为没有可学习参数
    
    - nn.Dropout(p=0.5, inplace=False)
        其中p表示元素变为0的概率，也就是有多少个神经元变为0.
"""
layers = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=0, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=1))

        
"""子模型可以从model.modules(), model.named_modules()来获得，从而进行相关操作，比如初始化"""

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        xavier_init(m)


"""子模型可以嵌套，通过model.add_module(module_name, nn.Sequential(*layers))"""



#%% 创建tensorflow版本的module



#%% 创建caffe版本的module


