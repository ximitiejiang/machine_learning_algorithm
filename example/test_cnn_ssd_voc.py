#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:38:55 2019

@author: ubuntu
"""
from utils.prepare_training import get_config, get_logger, get_dataset
from utils.runner import Runner
from torch.utils.data import DataLoader
from core.detector_lib import OneStageDetector
from core.batch_processor import batch_processor
from functools import partial
from model.parallel.collate import collate

# %% 模型训练
def train(cfg_path):
    """训练demo"""
    # 获得配置信息
    cfg = get_config(cfg_path)
    
    # 创建logger
    logger = get_logger(cfg.log_level)
    logger.info("start training:")
    
    # 创建模型
    model = OneStageDetector(cfg)
    model.to(cfg.device)
    
    # 创建数据
    dataset = get_dataset(cfg.data.train)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch_size,
                            sampler=cfg.sampler,
                            num_workers=cfg.num_workers,
                            collate_fn=partial(collate, samples_per_gpu=cfg.data.imgs_per_gpu),
                            pin_memory=False)
    
    # 创建训练器并开始训练
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir, cfg.log_level)
    runner.register_hooks()
    runner.run(dataloader)


# %% 模型测试
def test(cfg_path, imgs_path):
    """"测试demo"""
    pass
    


# %% 运行调试
if __name__ == "__main__":
    
    op = "train"
    
    if op=="train": 
        cfg_path = "cfg_ssd300_vgg16_voc.py"
        train(cfg=cfg_path)
    
    if op=="test":
        cfg_path = "cfg_ssd300_vgg16_voc.py"
        imgs_path = ""
        test(cfg_path, imgs_path)
        
        