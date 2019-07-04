#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:08:27 2019

@author: ubuntu
"""
from .base_model import BaseModel
import numpy as np
import math

class DTStump:
    """DTStump代表decision tree stump也就是决策树树桩(树桩代表只有一个root和2个leaf)"""
    def __init__(self, feat_id=None, feat_value=None, polarity=1, 
                 min_error=None, alpha=None):
        self.feat_id = feat_id
        self.feat_value = feat_value  
        self.polarity = polarity   # 存放预测样本正负的方式(polarity=1则小于value为负样本大于value为正样本)
        self.min_error = min_error
        self.alpha = alpha        # 该分类器的权重：误差越小，alpha越大，对应该分类器在预测中越重要


class LogisticStump:
    """尝试用logistic reg分类器来做基础弱分类器(作为弱分类器需要足够差异性)"""
    def __init__():
        pass

CLF_dict = {'DTStump': DTStump,
            'LogisticStump': LogisticStump}


class AdaBoost(BaseModel):
    
    def __init__(self, feats, labels, n_clfs, clf_type='DTStump'):
        """Ada boost分类器算法特点：采用最简单的只有一层的树作为基础分类器
        (也就是depth=1, root只进行了一次分解得到2个叶子节点)
        Args:
            feats(n_sample, n_feat)
            labels(n_sample,):  二分类，取值(-1,1)
        """
        super().__init__(feats, labels)
        # assert labels are [-1, 1]
        label_set = set(np.array(self.labels).flatten().astype(np.int8).tolist())
        for label in label_set:
            if label != 1 and label != -1:
                raise ValueError('labels should be 1 or -1.')
        
        self.n_clfs = n_clfs
        self.clf_list = []
        CLF = CLF_dict[clf_type]
        W = np.full((self.n_samples,), 1/self.n_samples)  # 初始化样本权值为1/N 
        # 创建n个分类器
        for clf_id in range(self.n_clfs):
            min_error = 1
            for feat_id in range(self.n_feats):  # 遍历每个特征的每个特征值
                feat = feats[:, feat_id]
                feat_unique = np.unique(feat)
                for value in feat_unique:
                    polarity = 1
                    preds = np.ones((self.n_samples,)) # 初始化单个分类器的预测值为1
                    preds[feat < value] = -1           # 在polarity=1条件下，小于阈值则输出预测为-1, 而polarity=-1时，则是小于阈值输出预测1，乘以polarity之后就统一标准为小于阈值输出-1。
                    error = sum(W[preds != labels])    # 计算误差率(w的创建和更新都做了概率化处理，sum(wi)=1,所以可以用错分样本的w之和作为误差率)
                    
                    if error > 0.5:
                        polarity = -1
                        error = 1 - error
                    
                    if error < min_error:   # 循环寻找最佳特征和特征值
                        min_error = error
                        best_polarity = polarity
                        best_feat_id = feat_id
                        best_value = value
                        best_preds = preds * polarity  # 乘以polarity后就是真实的预测结果
                        
            # 生成1个分类器
            alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10)) 
            clf = CLF(feat_id = best_feat_id,
                      feat_value = best_value,
                      polarity = best_polarity,
                      min_error = min_error,
                      alpha = alpha)
            self.clf_list.append(clf)
            print("finish no. %d stump build, acc is: %.3f, alpha is: %.5f"%(clf_id, 1 - clf.min_error, alpha))
            # 获取该分类器的预测结果后更新样本参数w
            W *= np.exp(-clf.alpha * labels * best_preds)  # 根据该预测值更新权重
            W /= np.sum(W) # 概率化，使w的值类似与概率分布值，在(0,1)之间，且所有w之和为1
    
        # 保存参数
        self.model_dict['model_name'] = 'AdaBoost Classifier' + '_'  \
                                        + clf_type + '_' + str(self.n_clfs) \
                                        + 'clfs'
        self.model_dict['clf_list'] = self.clf_list
                                         
                    
    def predict_single(self, sample):
        """单样本预测"""
        result = 0
        for clf in self.clf_list:
            if sample[clf.feat_id] < clf.feat_value:
                pred = -1 * clf.polarity
            else:
                pred = 1 * clf.polarity
            result += pred * clf.alpha
        result = np.sign(result)
        return result
            
        
        