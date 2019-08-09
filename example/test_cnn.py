#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:32:50 2019

@author: ubuntu
"""

from core.cnn_lib import Alexnet


if __name__ == "__main__":


# %% 准备数据
    train_dataset = datasets.CIFAR10(
            root= ""
            train=True,
            transform = transform.Compose([]))    
    val_dataset = datasets.CIFAR10(
            root= ""
            train=False,
            transform = transform.Compose([]))
    train_loader = DataLoader(
            train_dataset,
            batch_size
            shuffle
            sampler
            num_workers)
    val_loader = DataLoader(
            val_dataset,
            batch_size
            shuffle
            sampler
            num_workers)

# %% 准备模型    
    model = Alexnet()
    
# %% 准备训练
    