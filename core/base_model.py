#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:50:19 2019

@author: ubuntu
"""
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
import pickle
import os

class BaseModel():
    """ core function:
            model.train()
            model.evaluation()
            model.classfify()
            model.save()
            model.load()
    """
    def __init__(self, feats, labels):
        assert feats.ndim ==2, 'the feats should be (n_samples, m_feats), each sample should be 1-dim flatten data.'
        self.feats = feats
        self.labels = labels.astype(np.int8)
        self.trained = False
        
        self.feats = scale(self.feats)
    
    def get_batch_data(self, feats, labels, batch_size=16, type='shuffle'):
        """从特征数据中提取batch size个特征，并组合成一个特征数据
        """
        batch_idx = np.random.permutation(np.arange(len(labels)))[:batch_size]  # 随机出batch_size个idx
        batch_feats_list = []
        batch_labels_list = []
        for idx in batch_idx:
            batch_feats_list.append(feats[idx].reshape(1,-1))
            batch_labels_list.append(labels[idx].reshape(-1,1))
        batch_feats = np.concatenate(batch_feats_list, axis=0)
        batch_labels = np.concatenate(batch_labels_list, axis=0)
        return batch_feats, batch_labels
    
    def train(self, feats, labels):
        """训练函数，需要实现"""
        raise NotImplementedError('the classify func is not implemented.')
    
    def classify(self, single_sample_feats):
        """单样本分类函数，需要实现"""
        raise NotImplementedError('the classify func is not implemented.')
    
    def evaluation(self, test_feats, test_labels):
        """评价整个验证数据集
        """
        test_feats = scale(test_feats)
        
        correct = 0
        total_sample = len(test_feats)
        start = time.time()
        for feat, label in zip(test_feats, test_labels):
            pred_label, pred_prob = self.classify(feat)
            if int(pred_label) == int(label):
                correct += 1
        acc = correct / total_sample
        print('evaluation finished, with %f seconds.'%(time.time() - start))
        
        if test_feats.shape[1]==2: # 还没添加首列1，为2个特征
            self.vis_points_line(test_feats, test_labels, self.W)
        return acc
    
    def vis_loss(self, losses):
        """可视化损失"""
        assert losses is not None, 'can not visualize losses because losses is empty.'
        x = np.array(losses)[:,0]
        y = np.array(losses)[:,1]
        plt.subplot(1,2,1)
        plt.title('losses')
        plt.plot(x,y)
    
    def vis_points_line(self, feats, labels, W):
        """可视化二维点和分隔线(单组w)
        """
        assert feats.shape[1] == 2, 'feats should be 2 dimention data with 1st. column of 1.'
        assert len(W) == 3, 'W should be 3 values list.'
        
        feats_with_one = np.concatenate([np.ones((len(feats),1)), feats], axis=1)
        
        plt.subplot(1,2,2)
        plt.title('points and divide hyperplane')
        color = [c*64 + 64 for c in labels.reshape(-1)]
        plt.scatter(feats_with_one[:,1], feats_with_one[:,2], c=color)
        
        min_x = int(min(feats_with_one[:,1]))
        max_x = int(max(feats_with_one[:,1]))
        x = np.arange(min_x - 1, max_x + 1, 0.1)
        y = np.zeros((len(x),))
        for i in range(len(x)):
            y[i] = (-W[0,0] - x[i]*W[1,0]) / W[2,0]
        plt.plot(x, y, c='r')
    
    def save(self, path='./'):
        if self.trained:
            time1 = datetime.datetime.now()
            path = path + 'softmax_reg_weight_' + datetime.datetime.strftime(time1,'%Y%m%d_%H%M%S')
            with open(path, 'w') as f:
                pickle.dump(self.W, f)
            
    def load(self, path):
        if os.path.isfile(path):
            with open(path, 'r') as f:
                self.W = pickle.load(f)
        self.trained = True