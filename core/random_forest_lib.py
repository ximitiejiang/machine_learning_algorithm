#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:59:21 2019

@author: suliang
"""

from .decision_tree_lib import BaseTree, CARTClf
from .softmax_reg_lib import SoftmaxReg
import numpy as np
import time

CLF_dict = {'CARTClf': CARTClf,
            'SoftmaxReg': SoftmaxReg}

class RandomForest(BaseTree):
    
    def __init__(self, feats, labels, n_clfs, clf_type='CARTClf',
                 min_samples_split=2, max_depth=10, min_impurity_reduction = 1e-7,
                 max_features=None, sub_samples_ratio=None):
        """随机森林分类算法特点：随机取样本生成森林，属于2种集成学习方法(bagging和boosting)中的bagging
        - bagging：代表是random forest，对数据集进行有放回抽样，形成多个训练数据的子集，并在每个子集训练一个分类器，
          然后多个分类器的结果投票决定预测结果。
        - boosting：代表是adboost和GBDT，对数据样本进行加权，并多轮训练一个基本分类器的分类误差率来更新训练数据集的权重，
          同时更新分类器的系数，最后对基本分类器进行线性组合得到最终的一个分类器。
         - bagging与boosting的区别：bagging是多个分类器的组合，boosting是单个分类器的增强。
        """
        super().__init__(feats, labels, min_samples_split, max_depth, min_impurity_reduction)
        self.n_clfs = n_clfs
        self.clf_type = clf_type

        self.max_features = max_features
        self.sub_samples_ratio = sub_samples_ratio
        if not self.max_features:
            self.max_features = int(np.sqrt(feats.shape[1]))   # 一般每棵树取特征个数为原特征的子集，可取sqrt(n_feats)，也可取log(n_feats)
        if not self.sub_samples_ratio:
            self.sub_samples_ratio = 0.5                  # 一般每棵树取原样本个数的一半
    
    def train(self):        
        subsets, feats_id_list = self.get_random_subsets(self.feats, 
                                                         self.labels, 
                                                         self.n_clfs, 
                                                         self.max_features,
                                                         self.sub_samples_ratio)
        # 构建n棵子树分类器
        self.clf_list = []
        self.clf_feats_id_list = []
        start = time.time()
        for clf_id in range(self.n_clfs):
            clf_class = CLF_dict[self.clf_type]
            clf = clf_class(*subsets[clf_id], 
                          min_samples_split=2, 
                          max_depth=10,
                          min_impurity_reduction = 1e-7).train()
            self.clf_list.append(clf)
            self.clf_feats_id_list.append(feats_id_list[clf_id])
        print('Finished building trees in %f seconds.'%(time.time() - start))
        self.trained = True
        # 保存模型参数
        self.model_dict['model_name'] = 'RandomForest' + '_' \
                                        + self.clf_type + '_' \
                                        + str(self.n_clfs) + 'trees_' \
                                        + str(self.max_features) + 'subfeats_' \
                                        + str(self.sub_samples_ratio * 100) + '%subsamples'
        self.model_dict['clf_list'] = self.clf_list
        self.model_dict['clf_feats_id_list'] = self.clf_feats_id_list
        return self
        
    @staticmethod
    def get_random_subsets(feats, labels, n_subsets, n_subfeats, sub_samples_ratio=0.5):
        """用于生成k个随机的子数据集: 随机打乱样本排序，然后抽取部分行和部分列
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
        n_sub_samples = int(n_samples * sub_samples_ratio)  # 默认抽样个数是原样本的0.5
                
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
    
    def majority_vote(self, labels):
        """投票方式获得最多类别的label作为叶节点的预测结果
        Args: labels()
        """
        labels_count_dict = {}
        for label in labels:
            labels_count_dict[label] = labels_count_dict.get(label, 0) + 1
        max_result = max(zip(labels_count_dict.values(), 
                             labels_count_dict.keys()))
        return max_result[1]
    
    def predict_single(self, sample):
        """单样本预测，但一个样本需要遍历一个森林的所有树
        """
        result_i = []
        for i in range(len(self.clf_list)):
            clf = self.clf_list[i]
            feats_id = self.clf_feats_id_list[i]
            sample_refined = sample[feats_id]   # 先对样本提取对应特征列
            
            result_i.append(clf.predict_single(sample_refined)) #单颗树结果
        result = self.majority_vote(result_i)  # 投票计算结果
        return result
    
    
    