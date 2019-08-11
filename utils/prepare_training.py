#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:50:43 2019

@author: ubuntu
"""
import sys, os
from addict import Dict
from importlib import import_module
import logging


def get_config(config_path="cfg_ssd_voc.py"):
    """从py文件中获取配置信息，py文件中需要为变量赋值形式数据，导入为Dict类型"""
    file_name = os.path.abspath(os.path.expanduser(config_path))  # 获得完整路径
    module_name = os.path.basename(file_name)[:-3] # 取名字
    dir_name = os.path.dirname(file_name)
    
    # 导入module时需要确保路径在sys.path下才能成功导入
    if(not dir_name in sys.path):
        sys.path.insert(0, dir_name)
        cfg_data = import_module(module_name)   # 导入后python中的module对象：python中的包就是文件夹，module就是文件，相当于导入整个py文件
        sys.path.pop(0)
        
    cfg_dict = {}
    for name, value in cfg_data.__dict__.items():  # 从module对象中提取内部__dict__(python中一切皆对象，__dict__用来存放对象的属性，所以只要.py文件中都写成变量形式，就能在__dict__中找到)
        if not name.startswith("__"):
            cfg_dict[name] = value
    return Dict(cfg_dict)


def get_logger(log_level=logging.INFO):
    """创建logger"""
    # 先定义format/level
    format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format, level=log_level)
    # 再创建logger
    logger = logging.getLogger()
    return logger

def get_dataset():
    pass


if __name__ == "__main__":
    
    # 验证cfg: 但注意相对路径写法，需要相对于main
    cfg_path = "../example/cfg_ssd512_vgg16_voc.py"
    cfg = get_config(cfg_path)
    
    # 验证logger
    logger = get_logger(logging.INFO)
    logger.debug("debug")
    logger.info("info")
    
    # 验证数据集
    dataset = get_dataset()
    


