#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:22:57 2019

@author: ubuntu
"""
import numpy as np
import time, datetime
import pickle
import os
import math
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

class none_regularization():
    """没有正则化"""
    def __init__(self):
        pass
    
    def __call__(self, w):
        return 0
    
    def grad(self, w):
        return 0

class l1_regularization():
    """l1正则化"""
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, 1)
    
    def grad(self, w):
        return self.alpha * np.sign(w)

class l2_regularization():
    """l2正则化"""
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, w):
        return self.alpha * 0.5 * np.dot(w.T, w)
    
    def grad(self, w):
        return self.alpha * w   # 对原函数直接对w求导


class LinearRegression():
    """线性回归"""
    def __init__(self, feats, labels, lr=0.001, n_iters=200):
        
        self.feats = feats
        self.labels = labels
        self.n_feats = self.feats.shape[1]
        self.n_samples = self.feats.shape[0]
        
        self.n_iters = n_iters
        self.lr = lr
        self.regularization = none_regularization()
        self.trained = False
        self.model_dict = {}
        self.model_dict['model_name'] = 'LinearRegressor_with_no_regularization'
    
    def init_weights(self):
        limit = 1 / math.sqrt(self.n_feats + 1)  # 一列1也要算进去
        self.w = np.random.uniform(-limit, limit, (self.n_feats + 1, ))
    
    def train(self):
        # 在首列加1列全1
        feats_with_one = np.insert(self.feats, 0, 1, axis=1)
        self.init_weights()
        
        self.training_error = []
        for i in range(self.n_iters):
            y_pred = np.dot(feats_with_one, self.w)    # y' = wi * xi  (n,2)(2,)->(n,)
            mse = np.mean(0.5 * (self.labels - y_pred)**2 + self.regularization(self.w))  # mse = mean(0.5*(y - y')^2) 
            self.training_error.append(mse)
            grad = - np.dot((self.labels - y_pred), feats_with_one) + self.regularization.grad(self.w)  # grad = mse' = (y-y')*(-1)*(dy'/dw) = -(y-y')*x 
            self.w -= self.lr * grad
        # 绘制训练error    
        plt.figure()
        plt.plot(range(self.n_iters), self.training_error, label='mse')
        plt.title('mean square error in training')
        plt.legend()
        plt.grid()
        
        # 绘制训练完成后的预测曲线
        if self.n_feats == 1:
            self.evaluation(self.feats, self.labels, title='train')
        # 存放模型信息
        self.trained = True
        self.model_dict['w'] = self.w
        return self
        
    def predict_single(self, sample):
        """单样本预测：
        Args:
            sample: 
        """
        sample_with_one = np.insert(sample, 0, 1)
        y_pred = np.dot(sample_with_one, self.w)
        return y_pred
    
    def evaluation(self, test_feats, test_labels, title='test'):
        """对整个数据集进行预测，并评估均值平方误差"""
        feats_with_one = np.insert(test_feats, 0, 1, axis=1)
        y_preds = np.dot(feats_with_one, self.w)    # y' = wi * xi
        mse = np.mean(0.5 * (test_labels - y_preds)**2)
        print(title + ' mean square error for total datasets: %.3f'%mse)
        if test_feats.shape[1]==1:
            self.show_regress_curve(test_feats, test_labels, y_preds, title=title)
        return y_preds
    
    def show_regress_curve(self, feats, labels, y_preds, title=None):
        plt.figure()
        if title is None:
            title = 'regression curve'
        plt.title(title)
        plt.scatter(feats, labels, label='raw data', color='red')
        plt.plot(feats, y_preds, label = 'y_pred', color='green')
        plt.legend()
        plt.grid()
        
    def save(self, path='./demo/'):
        """保存模型，统一保存到字典model_dict中，但需要预先准备model_dict的数据
        """
        if self.trained and self.model_dict:  # 已训练，且model_dict不为空
            time1 = datetime.datetime.now()
            path = path + self.model_dict['model_name'] + '_' + datetime.datetime.strftime(time1,'%Y%m%d_%H%M%S') + '.pkl'
            with open(path, 'wb') as f:
                pickle.dump(self.model_dict, f)
        else:
            raise ValueError('can not save model due to empty model_dict or not trained.')
            
    def load(self, path=None):
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                self.model_dict = pickle.load(f)
        else:
            raise ValueError('model_dict does not existed in current path.')
        for key, value in self.model_dict.items():
            exec('self.' + key + '=value', {'self':self, 'value':value})
        self.trained = True

            
class RidgeRegression(LinearRegression):
    
    def __init__(self, feats, labels, reg_factor, lr=0.001, n_iters=200):
        """岭回归：采用l2正则化, reg_factor为正则化因子
        """
        super().__init__(feats=feats, labels=labels, lr=lr, n_iters=n_iters)
        self.regularization = l2_regularization(alpha = reg_factor)
        self.model_dict['model_name'] = 'RidgeRegressor_with_l2_regularization'
    
    
class LassoRegression(LinearRegression):
    
    def __init__(self, feats, labels, reg_factor, lr=0.001, n_iters=200):
        """lasso回归：采用l1正则化，reg_factor为正则化因子
        """
        super().__init__(feats=feats, labels=labels, lr=lr, n_iters=n_iters)
        self.regularization = l1_regularization(alpha = reg_factor)
        self.model_dict['model_name'] = 'LassoRegressor_with_l1_regularization'
        
    
# TODO
class PolynomialRegression(LinearRegression):
    
    def __init__(self, feats, labels, degree, lr=0.001, n_iters=200):
        """多项式回归：对特征进行多项式升维，得到非线性回归能力
        """
        feats_poly = self.polynomial_features(feats, degree)
        self.degree = degree
        super().__init__(feats=feats_poly, labels=labels, lr=lr, n_iters=n_iters)
        
    
    def polynomial_features(self, feats, degree):
        n_samples, n_features = np.shape(feats)
    
        def index_combinations():
            combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
            flat_combs = [item for sublist in combs for item in sublist]
            return flat_combs
        
        combinations = index_combinations()
        n_output_features = len(combinations)
        X_new = np.empty((n_samples, n_output_features))
        
        for i, index_combs in enumerate(combinations):  
            X_new[:, i] = np.prod(feats[:, index_combs], axis=1)
    
        return X_new    
    