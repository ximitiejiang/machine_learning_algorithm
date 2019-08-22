#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:10:04 2019

运行该文件，用于临时把repo根目录路径加入sys.path，从而在example文件夹中可以直接运行所有测试文件。

@author: ubuntu
"""
# %%
"""
注意：python运行特点，
    (1)main文件在哪里就会以哪个文件夹作为sys.path目录去进行寻找，
    这点可以打印sys.path可以看到每次在哪里运行main，这个路径就会自动被添加到sys.path
    
    (2)基于第一点，其他相对路径就必须要么在该main文件所在文件夹下面，要么在sys.path其他
    文件夹下面，能够拼接成完整路径。此时对于from model import xxx这种语句就容易找不到model
    文件夹的上级路径。必须把源码文件夹根路径作为统一的父路径添加到sys.path中。
    
    (3)还有很重要一点，不同版本的python对应了不同的sys.path和不同的PYTHONPATH。
    比如在命令行输入python是进入默认的python3.5，此时的sys.path，PYTHONPATH是一组。
    而输入python3进入的是python3.7，此时的sys.path和PYTHONPATH是另外一组
    这里需要指出，PYTHONPATH默认是空的，通过添加PYTHONPATH后，会自动导入到sys.path中
    
    (4)基于第2点，添加路径到sys.path的方法有：  
        1. 临时只针对当前终端
        sys.path.insert(0, path), 可以临时看到
        或者在命令行export PYTHONPATH=/home/xx/../xx  # 相当于添加一个临时变量，可通过env指令查看到
            
        2. 永久针对当前用户
        gedit ~/.bashrc                    # 这是打开用户目录~/下的bashrc文件
        export PYTHONPATH=/home/xx/../xx
        
        3. 永久针对所有用户
        sudo gedit /etc/profile   # 这是打开根目录/etc下的profile文件
        export PYTHONPATH=/home/xx/../xx
    
    但要注意：无论用上面那种方法，要搞清楚是添加到哪个python，一定要添加到自己在IDE中使用的那个python中去。
    (我添加总是会添加到python3.5，可我IDE用的是python3.7，所以始终不成功)
  
"""

import sys
import os

# %% 临时添加路径到sys.path中，但如果终端关闭则失效
path = os.path.abspath('..')  # 把当前main()函数路径的上上级路径添加进sys.path，也就是/xx/../machine_learning_algorithm 
if not path in sys.path:
    sys.path.insert(0, path)
