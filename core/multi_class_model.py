#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 08:54:55 2019

@author: ubuntu
"""

"""参考：https://blog.csdn.net/xfchen2/article/details/79621396
1. 多分类方式1：一对一
   流程：这是libsvm使用的方法，A对B类，A类对C类...B类对C类，B类对D类...C类对D类...分别训练(n-1)*(n-2)...*2个分类器，
   最后投票决定分类结果
   优点是：每个分类器都比较容易训练
   缺点是：如果类别较多会导致单分类器很多，训练和测试时间较长

2. 多分类方式2：一对多
   流程：A类对剩下B,C,D...B类对剩下A,C,D...,分别训练n_class个分类器，分别测试，对结果去最大值作为分类结果   
   优点是：子分类器个数相对较少，等于n_class个子分类器
   缺点是：每个分类器都是以全部样本训练，有多余的成分；
          同时负样本个数远远超过正样本个数，产生样本不平衡，需要引入不同的惩罚因子解决不平衡问题
"""
from .logistic_reg_lib import LogisticReg
from .svm_lib import SVMC
from .perceptron_lib import Perceptron
from .base_model import BaseModel

model_class_dict = {'logistic': 'LogisticReg',
                    'logistic_reg': 'LogisticReg',
                    'svm': 'SVMC',
                    'perceptron': 'Perceptron'}

class OVOModel(BaseModel):
    """多分类模型wrapper，采用one vs one的原理
    """
    def __init__(self, base_model, feats, labels, *args, **kwargs):
        assert base_model in model_class_dict.keys(), 'base_model name is not supported.'
        super().__init__(feats, labels)
        
        self.base_model = base_model
        self.n_classes = len(set(labels))
        self.n_samples = feats.shape[0]
        self.n_feats = feats.shape[1]
        self.model_dict = {}
        self.trained = False
        
        label_types = set(labels)
        ovo_feats = []
        ovo_labels = []
        
        self.ovo_models = []
        # 分解数据集
        for i, label in enumerate(label_types): 
            single_class_idx = labels[labels == label]  # TODO: 待debug这部分
            feats_single_class = feats[single_class_idx]
            labels_single_class = labels[single_class_idx]
            
            ovo_feats.append(feats_single_class)   # (n_clas, )
            ovo_labels.append(labels_single_class) # (n_clas, )
        # 创建模型
        self.n_models = self.factorial(self.n_classes - 1)  # ((n_class-1)!, )
        for j in range(self.n_models):
            model_class = model_class_dict[base_model]
            model = exec(model_class + '(feats, labels, *args, **kwargs)',
                         {'feats': ovo_feats[j],
                          'labels': ovo_labels[j],
                          'args': args, 
                          'kwargs': kwargs})
            self.ovo_models.append(model)
            
    @staticmethod
    def factorial(num):
        assert num > 0, 'num should bigger than 0.'
        factorial_op = 1
        for i in range(1, num+1):
            factorial_op *= i
        return factorial_op
        
    def train(self):
        """训练所有模型
        """
        for model in self.ovo_models:
            model.train()
            
        self.trained = True
        self.model_dict['model_name'] = 'ovo_model_' + self.ovo_models[0].model_dict['model_name']
        self.model_dict['ovo_models'] = self.ovo_models
    
    def predict_single(self, test_x):
        """单样本预测
        """
        assert self.trained, 'can not predict without trained weights.'
        result = []
        for model in self.ovo_models:
            result.append(model.predict_single(test_x))
        # vote
        predict = 1
        return predict
    
