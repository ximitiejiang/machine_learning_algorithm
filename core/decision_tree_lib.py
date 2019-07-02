#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:37:06 2019

@author: ubuntu
"""
import numpy as np
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
    

class CART(BaseModel):
    
    def __init__(self, feats, labels, 
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity = 1e-7):
        """CART分类树的特点是：基于gini指数的大小来识别最佳分隔特征，分割后gini指数越小说明这种
        分割方式更有利于数据的确定性提高。
        1. CART只会生成二叉树，因为判断时只判断等于特征值还是不等于(left分支是等于特征值分支，right分支是不等于特征值分支)
        2. gini计算只跟label相关，所以可以针对离散特征，也可以针对连续特征。不过对连续特征来说每个特征的取值个数非常多，
           计算量较大，可以考虑把连续特征离散化减少每个特征的value个数。
        """
        self.min_samples_split = min_samples_split  # 最少可分样本个数
        self.max_depth = max_depth                  # 最大树的深度
        self.min_gini = min_impurity                # 最小基尼指数，接近于0的一个值
        
        super().__init__(feats, labels)
        
#        self.dynamic_feat_id_list = list(np.arange(self.feats.shape[1]))  # 动态特征id，每分一次就去掉分过的feat id
        self.tree = self.create_tree(self.feats, self.labels, current_depth=0)
        
        # 没有训练，直接保存tree
        self.model_dict['model_name'] = 'CART classifier'\
            + '_depth' + str(self.tree_final_params['final_depth']) \
            + '_gini'+ str(round(self.tree_final_params['final_gini'], 3))
        self.model_dict['tree'] = self.tree
    
    def create_tree(self, feats, labels, current_depth=0):
        """创建CART树，存储特征
        """
        current_gini = self.calc_impurity(labels)  # 计算当前数据集的gini指数
        best_gini = 1
        best_criteria = None
        best_subsets = None
        
        n_samples, n_features = feats.shape
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feat_id in range(n_features): # 外循环提取特征列
                # 统计一个特征列的取值个数
                feat = feats[:, feat_id]
                feat_value_dict = self.count_quantity(feat)
                for value in feat_value_dict.keys(): # 内循环提取特征值
                    # 计算一个特征列的每个取值的gini: ge(great equal), lt(less than)
                    feats_ge, labels_ge, feats_lt, labels_lt = \
                    self.split_dataset(feats, labels, feat_id, value)
                    
                    gini = len(feats_ge) / len(feats) * self.calc_impurity(labels_ge) + \
                           len(feats_lt) / len(feats) * self.calc_impurity(labels_lt)
                           
                    if gini < best_gini and len(feats_ge) > 0 and len(feats_lt) > 0:
                        best_gini = gini
                        best_criteria = (feat_id, value)
                        best_subsets = [(feats_ge, labels_ge), (feats_lt, labels_lt)]   
        
        # 如果当前数据包的gini有下降且特征数大于1,则继续分解，此时存放中间节点，只有feat_id/feat_value/left/right
        if current_gini - best_gini > self.min_gini:  
#            self.dynamic_feat_id_list.pop(best_criteria[0])   # 弹出当前作为分割特征的列号feat_id不再作为分割选择
            return DTNode(feat_id = best_criteria[0], 
                          feat_value = best_criteria[1],
                          left = self.create_tree(*best_subsets[0], current_depth + 1),
                          right = self.create_tree(*best_subsets[1], current_depth + 1))
                        
        # 如果无法再分，此时存放叶子节点，只有points/labels/result
        labels_count_dict = self.count_quantity(labels)
        max_result = max(zip(labels_count_dict.values(), 
                             labels_count_dict.keys()))
        self.tree_final_params = dict(final_depth = current_depth + 1,
                                      final_gini = best_gini)  # 这里final_depth是包含叶子节点所以+1, 但不包含root节点
        return DTNode(points = feats,
                      labels = labels,
                      result = max_result[1])  # 存放label数量最多的标签值
                
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
        
    def calc_impurity(self, labels):
        """计算系统不纯度：在cart中用基尼指数来评价系统不纯度，只用于计算单个数据集按类别分的基尼(所以只传入labels即可计算)。
        该算法兼容离散特征和连续特征
        而多个数据集组合的基尼通过加权计算得到: 但注意加权系数在离散和连续特征中计算方法因为split_dataset改变而有区别。
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
            
class ID3(CART):
    def __init__(self, feats, labels, 
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity = 1e-7):
        super().__init__(feats, labels, min_samples_split, max_depth, min_impurity)
        self.min_info_gain = min_impurity

        
    def calc_impurity(self):
        """ID3算法采用信息增益来评估系统不纯度，信息增益 = 经验熵 - 条件熵
        """
        pass
    
    
class C45():
    def __init__(self, feats, labels, 
                 min_samples_split=2, 
                 max_depth=10,
                 min_impurity = 1e-7):
        super().__init__(feats, labels, min_samples_split, max_depth, min_impurity)
        self.min_gain_ratio = min_impurity
    
    def calc_impurity(self):
        """C45算法采用信息增益比来评估系统不纯度，信息增益比 = 信息增益 / 经验熵"""
        pass
        
        
        