#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:37:06 2019

@author: ubuntu
"""
import numpy as np
import math
from .base_model import BaseModel

class DTNode:
    """DTNode代表decision tree node"""
    def __init__(self, feat_id=None, feat_value=None, points=None, labels=None, 
                 right=None, left=None, result=None):
        self.feat_id = feat_id
        self.feat_value = feat_value  
        self.right = right 
        self.left = left
        self.points = points  # (optional for leaf)
        self.labels = labels  # (optional for leaf)
        self.result = result   # 存放最终叶子节点的result
    
class BaseTree(BaseModel):
    
    def __init__(self, feats, labels, 
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity_reduction = 1e-7):
        """分类和回归树的基类：创建决策树的核心过程在这个基类实现
                   ____BaseModel_______
                  /           \        \
             __CARTClf_       CARTReg  ada_boost
            /  |   \   \         \
          ID3 C45 RandF XGBoost   GBDT
        """
        super().__init__(feats, labels)
        self.min_samples_split = min_samples_split  # 最少可分样本个数
        self.max_depth = max_depth                  # 最大树的深度
        self.min_impurity_reduction = min_impurity_reduction # 最小系统不纯度缩减量，接近于0的一个值        
        self.tree = self.create_tree(self.feats, self.labels, current_depth=0)
    
    def calc_impurity_reduction(self):
        """统一用系统不纯度缩减量来描述系统：可以选择信息增益(也就是熵减少量,比如ID3)，或基尼减少量(比如CART)，
        或者均方值减少量(比如回归树)"""
        raise NotImplementedError('functions of clac_impurity or calc_leaf_value is not implemented.')
        
    def calc_leaf_value(self):
        raise NotImplementedError('functions of clac_impurity or calc_leaf_value is not implemented.')
    
    def create_tree(self, feats, labels, current_depth=0):
        """创建CART树，存储特征
        """
#        current_impurity = self.calc_impurity(labels)  # 计算当前数据集的gini指数
        best_impurity_reduction = 0
        best_criteria = None
        best_subsets = None
        
        n_samples, n_features = feats.shape
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feat_id in range(n_features): # 外循环提取特征列
                # 统计一个特征列的取值个数
                feat = feats[:, feat_id]
                feat_value_unique = np.unique(feat)
                for value in feat_value_unique: # 内循环提取特征值
                    # 计算一个特征列的每个取值的gini: ge(great equal), lt(less than)
                    feats_ge, labels_ge, feats_lt, labels_lt = \
                    self.split_dataset(feats, labels, feat_id, value)
                    

                    impurity_reduction = self.calc_impurity_reduction(labels, labels_ge, labels_lt)
                    
                    if impurity_reduction > best_impurity_reduction and len(feats_ge) > 0 and len(feats_lt) > 0:
                        best_impurity_reduction = impurity_reduction
                        best_criteria = (feat_id, value)
                        best_subsets = [(feats_ge, labels_ge), (feats_lt, labels_lt)]   
        
        # 如果当前数据包的gini有下降且特征数大于1,则继续分解，此时存放中间节点，只有feat_id/feat_value/left/right
        if best_impurity_reduction > self.min_impurity_reduction:  
            return DTNode(feat_id = best_criteria[0], 
                          feat_value = best_criteria[1],
                          left = self.create_tree(*best_subsets[0], current_depth + 1),
                          right = self.create_tree(*best_subsets[1], current_depth + 1))
                        
        # 如果无法再分，此时存放叶子节点，只有points/labels/result
        leaf_result = self.calc_leaf_value(labels)
        self.tree_final_params = dict(final_depth = current_depth + 1)  # 这里final_depth是包含叶子节点所以+1, 但不包含root节点
        return DTNode(points = feats,
                      labels = labels,
                      result = leaf_result)  # 存放label数量最多的标签值
    
    def split_dataset(self, feats, labels, feat_id, value):
        """分割指定的数据子集：小于value的一块，大于value的一块
        注意：这里跟lihang书上P70所述稍有差别，不是分成(等于value和不等于value)，
        而是分成(大于等于value和小于value)，因为这样改就可以兼容连续性特征操作。
        否则连续性特征等于value的子集往往只有1个数，会导致过拟合。
        (这种按大于等于/小于分割数据集的方法更普遍的应用在几个主要ML repo: 
        zhaozhiyong, eriklindernoren并用于其中的决策树/随机森林/xgboost算法中)
        Args:
            feats: 被分割的特征
            labels: 被分割的标签
            feat_id: 待分割的特征id
            value: 待分割的特征值
        """        
        feats_left = feats[feats[:, feat_id] >= value]
        labels_left = labels[feats[:, feat_id] >= value]
        feats_right = feats[feats[:, feat_id] < value]
        labels_right = labels[feats[:, feat_id] < value]
        return feats_left, labels_left, feats_right, labels_right
    
    def predict_single(self, sample):
        """单样本预测"""
        result = self.get_single_tree_result(sample, self.tree)
        return result
    
    def get_single_tree_result(self, sample, tree):
        """递归获得预测结果，用递归子函数是为了在predict_single()函数接口上去除tree这个变量
        同时该递归函数也是随机森林的基础函数
        """
        if tree.result != None:  # 当到达叶子节点，则直接返回tree.result作为预测标签
            return tree.result
        else:
            sample_value = sample[tree.feat_id]
            if sample_value >= tree.feat_value:  # 如果大于等于节点特征值，则进左树
                branch = tree.left
            else:                               # 如果不等于节点特征值，则进右树
                branch = tree.right
            return self.get_single_tree_result(sample, branch)
    

class CARTClf(BaseTree):
    
    def __init__(self, feats, labels, 
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity_reduction = 1e-7):
        """CART分类树的特点是：基于gini指数的大小来识别最佳分隔特征，分割后gini指数越小说明这种
        分割方式更有利于数据的确定性提高。
        1. CART只会生成二叉树，因为判断时只判断等于特征值还是不等于(left分支是等于特征值分支，right分支是不等于特征值分支)
        2. gini计算只跟label相关，所以可以针对离散特征，也可以针对连续特征。不过对连续特征来说每个特征的取值个数非常多，
           计算量较大，可以考虑把连续特征离散化减少每个特征的value个数。
        """     
        self.calc_impurity_reduction = self.calc_gini_reduction
        self.calc_leaf_value = self.majority_vote
        super().__init__(feats, labels,
                         min_samples_split=2, 
                         max_depth=10,
                         min_impurity_reduction = 1e-7)
        # 没有训练，直接保存tree
        self.trained = True
        self.model_dict['model_name'] = 'CARTClf'\
            + '_depth' + str(self.tree_final_params['final_depth'])
        self.model_dict['tree'] = self.tree
    
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
    
    def calc_gini_reduction(self, labels, labels_ge, labels_lt):
        """基尼缩减量可用于评估系统不纯度下降等级: 选择基尼缩减量最大的特征列和特征值
        基尼缩减量 = 系统分割前的gini - 系统分割后的gini = G(D) - (p1*gini(D1) + p2*gini(D2))
        (1-sum(pk^2)) - (D1/D*gini(D1) +D2/D*gini(D2) 
        """
        p = len(labels_ge) / len(labels)
        gini_split =  p * self.calc_gini(labels_ge) + \
                      (1 - p) * self.calc_gini(labels_lt)
        gini_reduction = self.calc_gini(labels) - gini_split
        return gini_reduction
    
    def calc_gini(self, labels):
        """计算系统不分割前的不纯度gini = 1-sum(p^2), p为每种类别标签的比例，也就是每种类别的概率
        """
        n_samples = labels.shape[0]
        if n_samples == 0:
            return 0
        label_dict = {}
        for label in labels:
            label_dict[label] = label_dict.get(label, 0) + 1
        gini = 1
        for value in label_dict.values():
            gini -= pow(value / n_samples, 2)
        return gini


class ID3Clf(CARTClf):
    def __init__(self, feats, labels, 
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity_reduction = 1e-7):
        """ID3分类算法：采用信息增益作为系统不纯度的评价标准"""
        self.calc_impurity = self.calc_info_gain
        super().__init__(feats, labels, min_samples_split, max_depth, min_impurity_reduction)
        
        # 没有训练，直接保存tree
        self.trained = True
        self.model_dict['model_name'] = 'ID3Clf'\
            + '_depth' + str(self.tree_final_params['final_depth'])
        self.model_dict['tree'] = self.tree
        
    def calc_info_gain(self, labels, labels_ge, labels_lt):
        """ID3算法，信息增益 = 经验熵 - 条件熵，条件熵是指H(Y|X)也就是分割后的熵
        info_gain = H(D) - (p1*H(D1) + p2*H(D2))
        """
        p = len(labels_ge) / len(labels)
        entropy_split = p * self.calc_entropy(labels_ge) + (1-p) * self.calc_entropy(labels_lt) 
        info_gain = self.calc_entropy(labels) - entropy_split
        return info_gain
    
    def calc_entropy(self, labels):
        """计算经验熵H=-sum(p*log(p,2)), p为每种类别标签的比例，也就相当于每种类别的概率, log为2为底
        """
        entropy = 0
        labels_unique = np.unique(labels)
        for label in labels_unique:
            p = len(labels[labels == label]) / len(labels)
            entropy += -p * math.log(p, 2)
        return entropy
            
    
class C45Clf(CARTClf):
    def __init__(self, feats, labels, 
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity_reduction = 1e-7):
        """C4.5分类算法：采用信息增益比作为系统不纯度的评价标准"""
        self.calc_impurity = self.calc_info_gain_ratio
        super().__init__(feats, labels, min_samples_split, max_depth, min_impurity_reduction)
        
        # 没有训练，直接保存tree
        self.trained = True
        self.model_dict['model_name'] = 'C45Clf'\
            + '_depth' + str(self.tree_final_params['final_depth'])
        self.model_dict['tree'] = self.tree
    
    def calc_info_gain_ratio(self):
        """C4.5算法，信息增益比 = 信息增益 / 经验熵"""
        pass
        
        
class CARTReg(BaseTree):
    """CART回归树算法，其建立树的方式跟分类树基本一样，唯2差别就是到叶子节点时决定叶子节点值采用取平均，而不是投票，
    因为回归的目的是尽可能靠近训练值的均值；同时在评估系统不纯度不是采用基尼指数，而是采用计算方差缩减(李航书P69)
    """
    def __init__(self, feats, labels,    
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity_reduction = 1e-7):
        self.calc_leaf_value = self.mean_of_y
        self.calc_impurity = self.calculate_variance_reduction
        super().__init__(feats, labels, 
                         min_samples_split=2, 
                         max_depth=10,
                         min_impurity_reduction = 1e-7)
        # 没有训练，直接保存tree
        self.model_dict['model_name'] = 'CARTReg'\
            + '_depth' + str(self.tree_final_params['final_depth']) \
            + '_variance'+ str(round(self.tree_final_params['final_impurity'], 3))
        self.model_dict['tree'] = self.tree
    
    def mean_of_y(self, y):
        """均值计算，用于在回归树中获得叶子节点的取值
        """
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]  # 均值计算：如果是one hot编码，返回多个平均，否则返回单个平均值
    
    def calculate_variance_reduction(self, y, y1, y2):
        """平方和缩减计算，用于评估系统做分割的特征和特征值是否最优
        """
        var = self.calculate_variance(y)
        var_1 = self.calculate_variance(y1)
        var_2 = self.calculate_variance(y2)
        p = len(y1) / len(y)
        # Calculate the variance reduction
        variance_reduction = var - (p * var_1 + (1 - p) * var_2)

        return sum(variance_reduction)

    
    def calculate_variance(self, X):
        """ Return the variance of the features in dataset X """
        mean = np.ones(np.shape(X)) * X.mean(0)   # (n_sample, n_feat) = 1
        n_samples = np.shape(X)[0]
        variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))  #
        
        return variance    
    