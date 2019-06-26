#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 07:35:43 2019

@author: ubuntu
"""

import numpy as np
from sklearn.preprocessing import scale
from .base_model import BaseModel
import time


class NaiveBayesContinuous(BaseModel):
    def __init__(self, feats, labels, norm):
        """ naive bayes algorithm lib, 朴素贝叶斯连续特征模型：要求特征都是连续性特征，不能有离散值
        代码参考：https://blog.csdn.net/u013597931/article/details/81705718(该代码结构清楚但在连续特征算法中遗漏了乘以先验概率)
        代码参考：https://blog.csdn.net/u013719780/article/details/78388056
        特点：没有参数，不需要训练，支持多分类，支持连续性特征
        
        Args:
            feats(numpy): (n_samples, f_feats)
            labels(numpy): (n_samples,)
        """
        super().__init__(feats, labels, norm=norm)
        
        self.classes_mean_dict, self.classes_std_dict = self.calc_mean_std()  
        # 模型参数准备
        self.model_dict['model_name'] = 'NaiveBayesContinuous'
        self.model_dict['classes_mean'] = self.classes_mean_dict
        self.model_dict['classes_std'] = self.classes_std_dict
        
    def calc_mean_std(self):
        """获得训练集的均值和方差，同时按类别分割数据集
        已知self.n_classes, self.classes_list, self.n_samples, self.n_feats
        Return:
            c_mean_dict: 按类别分的均值字典{class1: (n_feats, ), class2: (n_feats, ), ...}
            c_std_dict： 按类别分的标准差字典{class1: (n_feats, ), class2: (n_feats, ), ...}
            c_feat_dict：按类别分的特征字典{class1: (n_sample, n_feats), class2: (n_sample, n_feats, ), ...}
        """
        # 按类别分解feats
        c_feat_dict = {}
        for label in self.classes_list:
            temp_class_feat = np.array(self.feats[self.labels==label])
            c_feat_dict[label] = temp_class_feat
        #对每类特征的每种特征计算mean, std
        c_mean_dict = {}
        c_std_dict = {}
        for label in self.classes_list:
            c_mean = np.zeros((self.n_feats, ))  #(n_class, n_feats)
            c_std = np.zeros((self.n_feats, ))   #(n_class, n_feats)
            for i in range(self.n_feats):
                c_mean[i] = np.mean(c_feat_dict[label][:,i])   # 计算样本均值
                c_std[i] = np.std(c_feat_dict[label][:,i], ddof = 1) # 计算样本标准差的无偏估计(ddof表示除以n-1而不是n) 
            c_mean_dict[label] = c_mean
            c_std_dict[label] = c_std
        return c_mean_dict, c_std_dict
        
    def predict_single(self, sample_single):
        # 计算先验概率
        prior_prob_dict = self.calc_class_prior_prob()       # 先验概率P(Y=ck)     (n_class, )
#        x_condition_prob_group = np.zeros((self.n_classes, )) # 条件概率连乘P(Xi=xi|Y=ck)  (n_class, )
        probs_dict = {}
        for label in self.classes_list:            
            # 计算条件概率连乘
            single_class_mean = self.classes_mean_dict[label]
            single_class_std = self.classes_std_dict[label]
            x_condition_prob = self.calc_class_cond_prob(sample_single, 
                                                         single_class_mean, 
                                                         single_class_std)
            # 计算某一类别的P(Y=ck|X=x) = 先验概率P(Y=ck)*条件概率P(Xi=xi|Y=ck)
            probs_dict[label] = x_condition_prob * prior_prob_dict[label]
        # 预测
        _, best_label = sorted(zip(probs_dict.values(), probs_dict.keys()))[-1]
        
        return best_label
        
    def calc_class_prior_prob(self):
        """计算先验概率P(Y=ck)，即每种类别的概率
        """
        label_count = {}   # {label: count}
        for label in self.labels:
            label_count[label] = label_count.get(label, 0) + 1
        prior_prob_dict = {}   # {label: prob}
        total_count = len(self.labels)
        for label, count in label_count.items():
            prior_prob_dict[label] = count / total_count
        return prior_prob_dict
        
    def calc_class_cond_prob(self, sample, means, stds):
        """计算条件概率的连乘，即单个样本属于单个类的概率连乘P(Xi=xi|Y=ck)
        Args:
            sample(n_feats, )
            means(n_feats, )
            stds(n_feats, )
        """
        def gauss_probability(x, mean, std):
            """高斯概率的单值计算：用高斯概率密度计算出单点概率值作为该特征的概率
            这里是假定每个特征都是负责正态分布的，比如特征1为身高，则某个身高值1.75就可以知道他的概率密度是多少。
            由于常见特征大部分是正态分布，所以这里做了一个很强的假设。
            注意：通过高斯概率密度计算得到的是概率密度值，而不是概率值(有可能大于1), 但可用来进行相对大小的比较。
            """
            exponent = np.exp(- (np.power(x - mean, 2)) / (2 * np.power(std, 2)))
            gauss_prob = (1 / (np.sqrt(2 * np.pi) * std)) * exponent
            return gauss_prob
            
        x_condition_prob = 1   # x 代表条件概率的连乘
        for i in range(len(sample)):
            x_condition_prob *= gauss_probability(sample[i], means[i], stds[i])  # 计算一个特征在一个类别中的条件概率的连乘
        
        return x_condition_prob
    

class NaiveBayesDiscrete(NaiveBayesContinuous):
    def __init__(self):
        """ naive bayes algorithm lib, 朴素贝叶斯离散特征模型：要求特征都是离散性特征，不能有连续值
        """
        super().__init__()
        # 模型参数准备
        self.model_dict['model_name'] = 'NaiveBayesDiscrete'
    
    def predict_single(self, sample_single):
        # 计算先验概率
        prior_prob_dict = self.calc_class_prior_prob()       # 先验概率P(Y=ck)     (n_class, )
#        x_condition_prob_group = np.zeros((self.n_classes, )) # 条件概率连乘P(Xi=xi|Y=ck)  (n_class, )
        probs_dict = {}
        for label in self.classes_list:            
            # 计算条件概率连乘
            single_class_mean = self.classes_mean_dict[label]
            single_class_std = self.classes_std_dict[label]
            x_condition_prob = self.calc_class_cond_prob(sample_single, 
                                                         single_class_mean, 
                                                         single_class_std)
            # 计算某一类别的P(Y=ck|X=x) = 先验概率P(Y=ck)*条件概率P(Xi=xi|Y=ck)
            probs_dict[label] = x_condition_prob * prior_prob_dict[label]
        # 预测
        _, best_label = sorted(zip(probs_dict.values(), probs_dict.keys()))[-1]
        
        return best_label
    
    def calc_class_cond_prob(self, sample, means, stds):
        """离散型数据单类别条件概率的计算
        """
        # TODO: need finish.
        pass