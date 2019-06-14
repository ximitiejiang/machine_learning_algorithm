#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt
import time, datetime, os
import pickle
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

class Perceptron():
    def __init__(self, feats, labels):
        """ perceptron algorithm lib, 感知机模型
        特点：支持二分类，模型参数可保存(n_feat+1, 1)，支持线性可分数据
        
        softmax reg算法由一个线性模块(w0x0+w1x1+..wnxn)和一个非线性模块(softmax函数)组成一个函数
        用输入特征feats和labels来训练这个模块，得到一组(w0,w1,..wn)的模型，可用来进行二分类问题的预测，但不能直接用于多分类问题
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)  注意，多分类样本标签必须是0-n，从0开头因为算法中需要用label从0开始定位取对应概率值
        """
        assert feats.ndim ==2, 'the feats should be (n_samples, m_feats), each sample should be 1-dim flatten data.'
        self.feats = feats       
        self.labels = labels.astype(np.int8)
        self.trained = False
        
        # normalize, mnist特征取值范围(0-255), digits特征取值范围(0-16)，
        # 其中mnist由于数值较大会导致exp操作发生inf(无穷大)，所以需要先对特征进行normalize
        self.feats = scale(self.feats)  # to N(0,1)
    
    def get_batch_data(self, feats, labels, batch_size=16, type='shuffle'):
        """从特征数据中提取batch size个特征，并组合成一个特征数据
        """
        batch_idx = np.random.permutation(np.arange(len(labels)))[:batch_size]  # 随机出batch_size个idx
        batch_feats_list = []
        batch_labels_list = []
        for idx in batch_idx:
            label = 2 * labels[idx][0] - 1   # perceptron label(0, 1) to label(-1, 1)
            batch_feats_list.append(feats[idx].reshape(1,-1))
            batch_labels_list.append(np.array([label]).reshape(-1,1))
        batch_feats = np.concatenate(batch_feats_list, axis=0)
        batch_labels = np.concatenate(batch_labels_list, axis=0)
        return batch_feats, batch_labels
        
    def train(self, alpha=0.001, n_epoch=500, batch_size=2):
        """feats(x1,x2,..xn) -> feats(1,x1,x2,..xn)
        Args:
            alpha(float): 梯度下降步长
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
        sum_loss = 0
        sum_gradient = 0
        n_wrong = 0
        for i in range(n_iter):
            batch_feats, batch_labels = self.get_batch_data(
                feats_with_one, labels, batch_size=batch_size, type='shuffle')
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn, 然后通过softmax函数转换为概率(0-1)
            # (n_sample, 1) 每个样本一个prob(0~1)，也就是作为2分类问题的预测概率
            w_x = np.dot(batch_feats, self.W)       # w*x (b, 1)
            for j in range(len(w_x)):
                if w_x[j, 0] > 0:
                    continue
                else:
                    n_wrong += 1
                    sum_loss += - batch_labels[j, 0] * w_x[j, 0]  # loss = -sum(yi*wTx)
                    sum_gradient += - batch_labels[j, 0] * np.sum(batch_feats[j])
                    
            loss = sum_loss / n_wrong
            self.losses.append([i, loss])        # loss平均化  
            
            # vis text
            if i % 20 == 0:  # 每20个iter显示一次
                print('iter: %d / %d, loss: %f, '%(i, n_iter, loss))
            # gradient
            gradient = sum_gradient / n_wrong
            # update weight
            self.W -= alpha * gradient
            
        self.vis_loss(self.losses)
        if self.feats.shape[1] == 2:
            for j in range(self.W.shape[1]):   # w (3,4)代表3个特征，4个类别
                w = self.W[:,j].reshape(-1,1)
                self.vis_points_line(self.feats, self.labels, w)
        
        self.trained = True
        print('training finished, with %f seconds.'%(time.time() - start))
        
    def classify(self, single_sample_feats):
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
        """
        test_feats = scale(test_feats)   # 测试数据集跟训练数据集一样增加归一化操作
        
        correct = 0
        total_sample = len(test_feats)
        start = time.time()
        for feat, label in zip(test_feats, test_labels):
            pred_label, pred_prob = self.classify(feat)
            if int(pred_label) == int(label):
                correct += 1
        acc = correct / total_sample
        print('evaluation finished, with %f seconds.'%(time.time() - start))
        return acc
    
    def vis_loss(self, losses):
        """可视化损失, losses为list [(epoch, loss)]"""
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
            pickle.dump(self.W)
            
    def load(self, path):
        if os.path.isfile(path):
            self.W = pickle.load(path)
        self.trained = True

if __name__ == '__main__':

    import pandas as pd
    from sklearn.model_selection import train_test_split
    filename = '2classes_data_2.txt'  # 一个简单的2个特征的多分类数据集
    data = pd.read_csv(filename, sep='\t').values
    x = data[:,0:2]
    y = data[:,-1]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    
    perc = Perceptron(train_x, train_y)
    perc.train(alpha=0.5, n_epoch=10000, batch_size=64)  # 在学习率0.5下精度在0.8-0.9之间，太小学习率导致精度下降
    print('W = ', svm.W)
    acc = svm.evaluation(test_x, test_y)
    print('acc on test data is: %f'% acc)
    
    sample = np.array([2,8])
    label, prob = svm.classify(sample)
    print('one sample predict label = %d, probility = %f'% (label, prob))
