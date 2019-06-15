#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""

import numpy as np
from base import BaseModel
import time

class SVMReg(BaseModel):
    def __init__(self, feats, labels):
        """ svm reg algorithm lib, 可用于多分类的算法
        svm reg算法由一个线性模块(w0x0+w1x1+..wnxn)
        用输入特征feats和labels来训练这个模块，得到一组(w0,w1,..wn)的模型，可用来进行二分类问题的预测，但不能直接用于多分类问题
        Args:
            feats(numpy): (n_samples, m_feats)
            labels(numpy): (n_samples,)  注意，多分类样本标签必须是0-n，从0开头因为算法中需要用label从0开始定位取对应概率值
        """
        super().__init__(feats, labels)
    
    def softmax(self, x):
        """exp(wx)/sum(exp(wx))
        Args:
            x(array): (n_sample, k_class)
        Return:
            x_prob(array): (n_sample, k_class)
        """
        x_exp = np.exp(x)                    # (135,4), 这里要防止无穷大的产生
        x_sum = np.sum(x_exp, axis=1)        # (135,)
        x_sum_repeat = np.tile(x_sum.reshape(-1,1), (1, x.shape[1]))
        x_prob = x_exp / x_sum_repeat
        return x_prob
    
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
    

    
    def SMOsimple(self, data, labels, C, toler, maxIter):
        
        def selectJrand(i,m):
            j = i
            while(j==i):
                j = int(np.random.uniform(0,m))
            return j
        
        data = np.mat(data)                      # (100,2)
        labels = np.mat(labels).transpose()      # (100,1)
        m = data.shape[0]
        alphas = np.mat(np.zeros((m,1)))         # (100,1)
        b = 0
        
        iter = 0
        while (iter < maxIter):
            alphaPairsChanged = 0
            for i in range(m):
                fxi = float(np.multiply(alphas, labels).T * \
                      (data * data[i,:].T)) + b
                Ei = fxi - labels[i]
                # 判断所选alphaI是否为支持向量：alphaI>0, alphaI<C，则为支持向量
                # 判断alphaI对应的fxi的误差是否超过所定义偏差，如果超过说明需要优化alpha值
                if ((labels[i]*Ei < -toler) and (alphas[i]< C)) \
                   or ((labels[i]*Ei > toler) and (alphas[i]>0)): 
                    
                    # optimize step1: define alphaIold, alphaJold
                    j = selectJrand(i,m)
                    fxj = float(np.multiply(alphas, labels).T * \
                          (data * data[j,:].T)) + b
                    Ej = fxj - labels[j]
                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()
                    
                    # optimize step2: calculate L, H, eta
                    if (labels[i]==labels[j]):
                        L = max(0, alphas[j] +alphas[i] - C)
                        H = min(C, alphas[j] +alphas[i])
                    else:
                        L = max(0, alphas[j] -alphas[i])
                        H = min(C, C +alphas[j] -alphas[i])
                    if L==H:
                        continue
                    eta = 2.0*data[i,:]*data[j,:].T - data[i,:]*data[i,:].T - \
                          data[j,:]*data[j,:].T
                    if eta >=0:
                        continue
                    
                    # optimize step3: update alphaInew, alphaJnew
                    alphas[j] -= labels[j]*(Ei - Ej)/eta 
                    if alphas[j] > H:
                        alphas[j] = H
                    elif alphas[j]< L:
                        alphas[j] = L
                    if ((alphas[j] - alphaJold)<0.00001):
                        continue
                    alphas[i] += labels[i]*labels[j]*(alphaJold-alphas[j])
                    
                    # optimize step4: update b
                    b1 = b - Ei - labels[i]*(alphas[i]-alphaIold)* \
                         data[i,:]*data[i,:].T - labels[j]*(alphas[j]-alphaJold)*\
                         data[i,:]*data[j,:].T
                    b2 = b - Ej - labels[i]*(alphas[i]-alphaIold)* \
                         data[i,:]*data[j,:].T - labels[j]*(alphas[j]-alphaJold)*\
                         data[j,:]*data[j,:].T
                    if alphas[i] > 0 and alphas[i] < C:
                        b = b1
                    elif alphas[j] > 0 and alphas[j] < C:
                        b = b2
                    else:
                        b = 0.5*(b1 + b2)
                        
                    alphaPairsChanged +=1
                    
            if (alphaPairsChanged ==0): # 如果不再有alpha进行优化，就最多运行MaxIter次
                iter += 1
            else:
                iter = 0                # 如果有alpha优化过，则重新计算循环次数
    
        return alphas, b    
    
    def train(self, alpha=0.001, n_epoch=500, batch_size=16):
        """feats(x1,x2,..xn) -> feats(1,x1,x2,..xn)
        Args:
            alpha(float): 梯度下降步长
            n_epoch(inf): 循环训练轮数
        """
        assert batch_size <= len(self.labels), 'too big batch size, should be smaller than dataset size.'
        
        start = time.time()
        n_classes = len(set(self.labels))
        n_samples = len(self.feats)
        feats_with_one = np.concatenate([np.ones((n_samples,1)), self.feats], axis=1)  # (n_sample, 1+ n_feats) (1,x1,x2,..xn)
        labels = self.labels.reshape(-1, 1)   # (n_sample, 1)
        self.W = np.zeros((feats_with_one.shape[1], n_classes)) # (f_feat, c_classes)
        self.losses = []
        
        n_iter = n_epoch * (n_samples // batch_size)
        for i in range(n_iter):
            batch_feats, batch_labels = self.get_batch_data(
                feats_with_one, labels, batch_size=batch_size, type='shuffle')
            
            # 基于feats, labels，采用SMO算法算出alpha向量
            alpha, b = self.SMO_simple(batch_feats, batch_labels, 
                                  C = 100, toler = 0.001, maxIter = 40)
            
            # w0*x0 + w1*x1 +...wn*xn = w0 + w1*x1 +...wn*xn, 然后通过softmax函数转换为概率(0-1)
            # (n_sample, 1) 每个样本一个prob(0~1)，也就是作为2分类问题的预测概率
            w_x = np.dot(batch_feats, self.W)       # w*x (b, c)
            probs = self.softmax(w_x)               # probability (b,c)
            
            # loss: 只提取正样本概率p转化为损失-log(p) 
            sum_loss = 0
            for sample in range(len(batch_labels)):
                sum_loss += -np.log(probs[sample, batch_labels[sample, 0]] / np.sum(probs[sample, :]))
            loss = sum_loss / len(batch_labels)  # average loss
            self.losses.append([i,loss])
            
            # vis text
            if i % 20 == 0 and i != 0:  # 每20个iter显示一次
                print('iter: %d / %d, loss: %f, '%(i, n_iter, loss))
            # gradient    
            _probs = - probs              # 取负号 -p
            for j in range(batch_size):
                label = batch_labels[j, 0]   # 提取每个样本的标签
                _probs[j, label] += 1   # 正样本则为1-p, 负样本不变依然是-p    
            gradient = - np.dot(batch_feats.transpose(), _probs)  # (135,3).T * (135,4) -> (3,4), grad = -x*(I-y')   
            # update Weights
            self.W -= alpha * gradient * (1/batch_size)   # W(3,4) - (a/n)*(3,4), 因前面计算梯度时我采用的负号，这里就应该是w = w-alpha*grad
        
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
        probs = self.softmax(np.dot(single_sample_feats, self.W))  # w*x
        label = np.argmax(probs)   # 获得最大概率所在位置，即标签(所以也要求标签从0开始 0~n)
        label_prob = probs[0,label]
        return label, label_prob
    

if __name__ == '__main__':

    import pandas as pd
    from sklearn.model_selection import train_test_split
    filename = '2classes_data_2.txt'  # 一个简单的2个特征的多分类数据集
    data = pd.read_csv(filename, sep='\t').values
    x = data[:,0:2]
    y = data[:,-1]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    
    svm = SVMReg(train_x, train_y)
    svm.train(alpha=0.5, n_epoch=10000, batch_size=64)  # 在学习率0.5下精度在0.8-0.9之间，太小学习率导致精度下降
    print('W = ', svm.W)
    acc = svm.evaluation(test_x, test_y)
    print('acc on test data is: %f'% acc)
    
    sample = np.array([2,8])
    label, prob = svm.classify(sample)
    print('one sample predict label = %d, probility = %f'% (label, prob))
