#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:59:21 2019

@author: suliang
"""

from decision_tree import CART
import numpy as np
from math import log

class RandomForest(CART):
    
    def __init__(self, feats, labels, n_trees):
        """随机森林分类算法特点：随机取样本生成森林，属于2种集成学习方法(bagging和boosting)中的bagging
        - bagging：代表是random forest，对数据集进行有放回抽样，形成多个训练数据的子集，并在每个子集训练一个分类器，
          然后多个分类器的结果投票决定预测结果。
        - boosting：代表是adboost和GBDT，对数据样本进行加权，并多轮训练一个基本分类器的分类误差率来更新训练数据集的权重，
          同时更新分类器的系数，最后对基本分类器进行线性组合得到最终的一个分类器。
         - bagging与boosting的区别：bagging是多个分类器的组合，boosting是单个分类器的增强。
        """
        super().__init__(feats, labels)
        self.n_trees = n_trees
        
    
    def train(self):
        n_subfeats = int(log(self.n_feats , 2)) + 1 if self.n_feats > 2 else 1 # 特征个数，其中k=log2(n_feats),
        n_subsets = self.n_trees         # 子集个数
        
        self.tree_list = []
        self.tree_feat_idx = []
        subsets, feats_id_list = self.get_random_subsets(self.feats, 
                                                        self.labels, 
                                                        n_subsets, 
                                                        n_subfeats)
        # 构建n棵树        
        for i in range(self.n_trees):
            tree = self.create_tree(*subsets[i])
            self.tree_list.append(tree)
            self.tree_feat_idx.append(feats_id_list[i])
                
        self.trained = True
        # 保存模型参数
        self.model_dict['model_name'] = 'RandomForest classifier'
        self.model_dict['tree_list'] = self.tree_list
        self.model_dict['tree_feat_idx'] = self.tree_feat_idx
        
    @staticmethod
    def get_random_subsets(feats, labels, n_subsets, n_subfeats):
        """用于生成k个随机的子数据集: 随机打乱样本排序，然后抽取所有样本
        Args:
            feats(n_sample, n_feats)
            labels(n_sample,)
            n_subsets: 要求的子样本个数
            n_subfeats: 每个子样本的特征个数
        Return:
            subsets(n): n个subsets
            feats_id_list: 每个subset对应的feat id
        """
        n_samples, n_feats = feats.shape[0], feats.shape[1]
        n_sub_samples = n_samples  # 默认抽样个数就是总样本个数
                
        subsets = []
        feats_id_list = []
        for _ in range(n_subsets):
            # 随机特征id
            feats_idx = np.random.choice(range(n_feats),
                                         size=n_subfeats)
            # 随机样本id
            samples_idx = np.random.choice(range(n_samples), 
                                          size=n_sub_samples) # 随机抽取样本个数
            # 获得特征列
            sub_feats = feats[:, feats_idx]
            # 获得样本行
            subsets.append([sub_feats[samples_idx], labels[samples_idx]])
            feats_id_list.append(feats_idx)
            
        return subsets, feats_id_list
        
    
    def choose_samples(self, k):
        """从基础数据中随机抽取包含k个特征的样本
        """
        pass
        
#    def predict_single(self, sample):
#        """单样本预测"""
#        def get_result(sample, tree):
#            """递归获得预测结果，用递归子函数是为了在predict_single()函数接口上去除tree这个变量"""
#            if tree.result != None:  # 当到达叶子节点，则直接返回tree.result作为预测标签
#                return tree.result
#            else:
#                sample_value = sample[tree.feat_id]
#                if sample_value == tree.feat_value:  # 如果等于节点特征值，则进左树
#                    branch = tree.left
#                else:                               # 如果不等于节点特征值，则进右树
#                    branch = tree.right
#                return get_result(sample, branch)
#            
#        result = get_result(sample, self.tree)
#        return result    
    
    
    def predict_single(self, sample):
        """单样本预测，但一个样本需要遍历一个森林的所有树
        """
        for i in range(len(self.tree_list)):
            tree = self.tree_list[i]
            feat_id = self.tree_feat_idx[i]
            sample_refined = sample[:, feat_id]
            
            result_i.append(self.get_single_tree_result(sample_refined, tree))
        
        result = np.sum(result_i)
        return result
    
    def evaluation(self):
        """所有样本的预测"""
        result = []
        for i in range(len(self.tree_list)):
            
    
    