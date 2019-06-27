#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:37:06 2019

@author: ubuntu
"""
import numpy as np
from .base_model import BaseModel

class Node:
    def __init__(self, feat_id=None, feat_value=None, points=None, labels=None, 
                 right=None, left=None, result=None):
        self.feat_id = feat_id
        self.feat_value = feat_value  
        self.right = right 
        self.left = left
        self.points = points  
        self.labels = labels
        self.result = result   # 存放最终叶子节点的result
    

class CART(BaseModel):
    
    def __init__(self, feats, labels, norm=False):
        super().__init__(feats, labels, norm=norm)
        
        self.dynamic_feat_id_list = list(np.arange(self.feats.shape[1]))  # 动态特征id，每分一次就去掉分过的feat id
        self.tree = self.create_tree(self.feats, self.labels)
    
    def create_tree(self, feats, labels):
        """创建CART树，存储特征
        """
        current_gini = self.calc_gini(labels)  # 计算当前
        best_gini = 1
        best_criteria = None
        best_subsets = None
        for feat_id in self.dynamic_feat_id_list: # 外循环提取特征列
            # 统计一个特征列的取值个数
            feat = feats[:, feat_id]
            feat_value_dict = self.count_quantity(feat)
            for value in feat_value_dict.keys(): # 内循环提取特征值
                # 计算一个特征列的每个取值的gini
                feats_yes, labels_yes, feats_no, labels_no = \
                self.split_dataset(feats, labels, feat_id, value)
                
                gini = len(feats_yes) / len(feats) * self.calc_gini(labels_yes) + \
                       len(feats_no) / len(feats) * self.calc_gini(labels_no)
                       
                if gini < best_gini and len(feats_yes) > 0 and len(feats_no) > 0:
                    best_gini = gini
                    best_criteria = (feat_id, value)
                    best_subsets = [(feats_yes, labels_yes), (feats_no, labels_no)]   
        
        if current_gini > 0 and len(self.dynamic_feat_id_list) > 1:  # 如果当前数据包的gini不等于0且特征数大于1,则继续分解，此时存放中间节点，只有feat_id/feat_value/left/right
            self.dynamic_feat_id_list.pop(best_criteria[0])   # 弹出当前作为分割特征的列号feat_id不再作为分割选择
            return Node(feat_id = best_criteria[0], 
                        feat_value = best_criteria[1],
                        left = self.create_tree(*best_subsets[0]),
                        right = self.create_tree(*best_subsets[1]))
                        
        else:  # 如果当前数据包的gini=0，则说明标签分类已经干净(同种标签)，此时存放叶子节点，只有points/labels
            return Node(points = feats,
                        labels = labels,
                        result = labels[0])
                
                
    def count_quantity(self, datas):
        """由于有多处要统计个数，统一写成一个函数
        Args:
            datas(n,)
        """
        count_dict = {}
        for value in datas:
            count_dict[value] = count_dict.get(value, 0) + 1
        
        return count_dict
    
    def split_dataset(self, feats, labels, feat_id, value):
        """分割指定的数据子集：小于value的一块，大于value的一块
        Args:
            feats: 被分割的特征
            labels: 被分割的标签
            feat_id: 待分割的特征id
            value: 待分割的特征值
        """
        feats_yes = feats[feats[:, feat_id] == value]
        labels_yes = labels[feats[:, feat_id] == value]
        feats_no = feats[feats[:, feat_id] != value]
        labels_no = labels[feats[:, feat_id] != value]
        return feats_yes, labels_yes, feats_no, labels_no
        
    def calc_gini(self, labels):
        """计算基尼指数：本部分只用于计算单个数据集按类别分的基尼(所以只传入labels即可计算)。
        而多个数据集组合的基尼通过加权计算得到。
         
        """
        n_samples = labels.shape[0]
        if n_samples == 0:
            return 0
        label_dict = self.count_quantity(labels)
        gini = 1
        for value in label_dict.values():
            gini -= pow(value / n_samples, 2)
        return gini

    def predict_single(self, sample):
        
        def get_result(sample, tree):
            """递归获得预测结果，嵌套这个递归是为了在predict_single()函数接口上去除tree这个变量"""
            if tree.result != None:  # 当到达叶子节点，则直接返回tree.result作为预测标签
                return tree.result
            else:
                sample_value = sample[tree.feat_id]
                if sample_value >= tree.feat_value:
                    branch = tree.right
                else:
                    branch = tree.left
                return get_result(sample, branch)
        result = get_result(sample, self.tree)
        return result
            
    
    
class ID3():
    def __init__(self):
        pass
    
class C45():
    def __init__(self):
        pass