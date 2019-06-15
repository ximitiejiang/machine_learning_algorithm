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

class Perceptron():
    def __init__(self, feats, labels):
        """ perceptron algorithm lib, 感知机模型
        特点：支持二分类，模型参数可保存(n_feat+1, 1)，支持线性可分数据
        
        perceptron算法只有一个线性模块(w0x0+w1x1+..wnxn)，该线性模块的输出正负就代表判定结果(>0为正样本，<0为负样本)
        通过输出y*(wTx)<0来判定错分样本，然后损失函数就定义为所有错分样本的函数间隔之和loss=-sum(y*(wTx))
        Args:
            feats(numpy): (n_samples, f_feats)
            labels(numpy): (n_samples,)
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
            if labels[idx] == 0:
                label = 2 * labels[idx][0] - 1   # perceptron label(0, 1) to label(-1, 1)
            else:
                label = labels[idx]
            batch_feats_list.append(feats[idx].reshape(1,-1))
            batch_labels_list.append(np.array([label]).reshape(-1,1))
        batch_feats = np.concatenate(batch_feats_list, axis=0)
        batch_labels = np.concatenate(batch_labels_list, axis=0)
        return batch_feats, batch_labels
        
    def train(self, alpha=0.001, n_epoch=500, batch_size=1):
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
            self.W -= alpha * gradient
            
        self.vis_loss(self.losses)
        if self.feats.shape[1] == 2:
            for j in range(self.W.shape[1]):
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
            pred_label = self.classify(feat)
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
    filename = '2classes_data_2.txt'  # 一个简单的2个特征的2分类数据集
    data = pd.read_csv(filename, sep='\t').values
    x = data[:,0:2]
    y = data[:,-1]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    
    perc = Perceptron(train_x, train_y)
    perc.train(alpha=0.01, n_epoch=10, batch_size=1)
    print('W = ', perc.W)
    acc = perc.evaluation(test_x, test_y)
    print('acc on test data is: %f'% acc)
    
    sample = np.array([2,8])
    label = perc.classify(sample)
    print('one sample predict label = %d'% label)
