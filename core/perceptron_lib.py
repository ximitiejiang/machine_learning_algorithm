#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""

import numpy as np
from sklearn.preprocessing import scale
from .base_model import BaseModel
import time

class Perceptron(BaseModel):
    def __init__(self, feats, labels):
        """ perceptron algorithm lib, 感知机模型
        特点：支持二分类，模型参数可保存(n_feat+1, 1)，支持线性可分数据
        
        perceptron算法只有一个线性模块(w0x0+w1x1+..wnxn)，该线性模块的输出正负就代表判定结果(>0为正样本，<0为负样本)
        通过输出y*(wTx)<0来判定错分样本，然后损失函数就定义为所有错分样本的函数间隔之和loss=-sum(y*(wTx))
        Args:
            feats(numpy): (n_samples, f_feats)
            labels(numpy): (n_samples,)
        """
        super().__init__(feats, labels)
        
        # assert labels are [-1, 1]
        label_set = set(self.labels)
        for label in label_set:
            if label != 1 and label != -1:
                raise ValueError('labels should be 1 or -1.')
    
    def get_batch_data(self, feats, labels, batch_size=16, type='shuffle'):
        """从特征数据中提取batch size个特征，并组合成一个特征数据
        """
        batch_idx = np.random.permutation(np.arange(len(labels)))[:batch_size]  # 随机出batch_size个idx
        batch_feats_list = []
        batch_labels_list = []
        for idx in batch_idx:
            if labels[idx] == 0:
                label = 2 * labels[idx][0] - 1   # perceptron label(0, 1) to label(-1, 1)
            else:
                label = labels[idx]
            batch_feats_list.append(feats[idx].reshape(1,-1))
            batch_labels_list.append(np.array([label]).reshape(-1,1))
        batch_feats = np.concatenate(batch_feats_list, axis=0)
        batch_labels = np.concatenate(batch_labels_list, axis=0)
        return batch_feats, batch_labels
        
    # TODO: 采用阶越函数代替在train和test中的判断
    def step(self, x):
        if x > 0:
            return 1
        else:
            return -1   
    
    def train(self, lr=0.001, n_epoch=500, batch_size=1):
        """feats(x1,x2,..xn) -> feats(1,x1,x2,..xn)
        Args:
            lr(float): 梯度下降步长
            n_epoch(inf): 循环训练轮数
        """                  
        assert batch_size <= len(self.labels), 'too big batch size, should be smaller than dataset size.'
        
        start = time.time()
        n_samples = len(self.feats)
        feats_with_one = np.concatenate([np.ones((n_samples,1)), self.feats], axis=1)  # (n_sample, 1+ n_feats) (1,x1,x2,..xn)
        labels = self.labels.reshape(-1, 1)   # (n_sample, 1)
        self.W = np.zeros((feats_with_one.shape[1], 1)) # (f_feat, 1)
        self.losses = []
        
        n_iter = n_epoch * (n_samples // batch_size)
        for i in range(n_iter):
            sum_loss = 0
            sum_gradient = np.zeros((len(self.W), 1))
            n_wrong = 0
            batch_feats, batch_labels = self.get_batch_data(
                feats_with_one, labels, batch_size=batch_size, type='shuffle')
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn
            w_x = np.dot(batch_feats, self.W)       # w*x (b, 1)
            for j in range(len(w_x)):
                prob = batch_labels[j, 0] * w_x[j, 0]  # 可以把函数间隔看成一个确信度(prob，参考李航书p97,svm对函数间隔的描述)
                if prob > 0:  # 正样本不处理
                    continue
                else:         # 负样本：累加wTx
                    n_wrong += 1
                    sum_loss += - prob  # loss = -sum(yi*wTx)
                    sum_gradient += (- batch_labels[j, 0] * batch_feats[j]).reshape(-1,1) # 累加一个batch size每个负样本产生的梯度
                    
            loss = sum_loss / n_wrong if n_wrong !=0 else 0
            self.losses.append([i, loss])        # loss平均化  
            # vis text
            if i % 20 == 0 and i != 0:  # 每20个iter显示一次
                print('iter: %d / %d, loss: %f, '%(i, n_iter, loss))
            # gradient
            gradient = sum_gradient / n_wrong if n_wrong !=0 else 0  # (f_feat+1, 1) 梯度平均
            # update weight
            self.W -= lr * gradient
            
        self.vis_loss(self.losses)
        if self.feats.shape[1] == 2:
            for j in range(self.W.shape[1]):
                w = self.W[:,j].reshape(-1,1)
                self.vis_points_line(self.feats, self.labels, w)
        
        self.trained = True
        print('training finished, with %f seconds.'%(time.time() - start))
     
    def predict_single(self, single_sample_feats):
        """ 单样本预测
        Args:
            data(numpy): (1, m_feats) or (m_feats,)，如果是(m,n)则需要展平
            k(int): number of neibours
        Returns:
            label(int)
        """
        assert isinstance(single_sample_feats, np.ndarray), 'data should be ndarray.'
        assert (single_sample_feats.shape[0]==1 or single_sample_feats.ndim==1), 'data should be flatten data like(m,) or (1,m).'
        assert self.trained, 'model didnot trained, can not classify without pretrained params.'
        single_sample_feats = np.concatenate([np.array([1]), single_sample_feats]).reshape(1,-1)
        result = np.sum(np.dot(single_sample_feats, self.W))  # w*x
        if result > 0:
            return 1
        else:
            return -1
    
    def evaluation(self, test_feats, test_labels):
        """评价整个验证数据集
        Args:
            test_feats
            test_labels: 输入(0-1)会转化为(-1,1)给perceptron算法
        """
        test_feats = scale(test_feats)   # 测试数据集跟训练数据集一样增加归一化操作
        test_labels_new = np.ones_like(test_labels)
        for idx in range(len(test_labels)):
            if test_labels[idx] == 0:
                test_labels_new[idx] = 2 * test_labels[idx] - 1   # perceptron label(0, 1) to label(-1, 1)
            else:
                test_labels_new[idx] = test_labels[idx]
        
        correct = 0
        total_sample = len(test_feats)
        start = time.time()
        for feat, label in zip(test_feats, test_labels_new):
            pred_label = self.predict_single(feat)
            if int(pred_label) == int(label):
                correct += 1
        acc = correct / total_sample
        print('evaluation finished, with %f seconds.'%(time.time() - start))
        return acc

