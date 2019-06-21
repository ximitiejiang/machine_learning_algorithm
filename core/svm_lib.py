#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:34:02 2019

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt
from .base_model import BaseModel

# TODO: 实现SVM的多分类版本：
"""参考：https://blog.csdn.net/xfchen2/article/details/79621396
1. 多分类方式1：一对一
   流程：这是libsvm使用的方法，A对B类，A类对C类...B类对C类，B类对D类...C类对D类...分别训练(n-1)*(n-2)...*2个分类器，最后投票决定分类结果
   优点是：每个分类器都比较容易训练
   缺点是：如果类别较多会导致单分类器很多，训练和测试时间较长

2. 多分类方式2：一对多
   流程：A类对剩下B,C,D...B类对剩下A,C,D...,分别训练n_class个分类器，分别测试，对结果去最大值作为分类结果   
   优点是：子分类器个数相对较少，等于n_class个子分类器
   缺点是：每个分类器都是以全部样本训练，有多余的成分；
          同时负样本个数远远超过正样本个数，产生样本不平衡，需要引入不同的惩罚因子解决不平衡问题
"""

class SVMC(BaseModel):
    """SVMClassify
    
    调试细节：
        1. 输入feat/label应该为mat格式，feat(n_sample, n_feat), label(n_sample, 1)
        2. label的输入必须是1,-1,这点跟perceptron算法一样
        3. 特征norm=False，否则无法收敛
    """
    def __init__(self, feats, labels, C, toler, max_iter, kernel_option=('rbf', 0.5)):
        assert isinstance(feats, np.matrix) and isinstance(labels, np.matrix), 'feats and labels should be mat.'
        assert labels.ndim ==2 and labels.shape[1] == 1, 'labels should be (n,1) style mat.'
        
        super().__init__(feats, labels, norm=False)
        
        self.feats = np.mat(self.feats)    # scale()函数会把feat从mat变为array，所以这里增加一次变换
        self.labels = np.mat(self.labels)
        
        self.C = C 
        self.toler = toler 
        self.max_iter = max_iter
        self.n_samples = np.shape(feats)[0] 
        self.alphas = np.mat(np.zeros((self.n_samples, 1)))        # (n_sample, 1)
        self.b = 0
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))     # (n_sample, 2) 2代表
        self.kernel_opt = kernel_option
        # 预先计算好核函数的输出数据
        self.kernel_mat = self.calc_kernel(self.feats, self.kernel_opt)  # 核函数的输出 (n_sample, n_sample)，每个样本跟其他样本都做一次升维和内积计算
        
        # assert labels are [-1, 1]
        label_set = set(np.array(self.labels).flatten().astype(np.int8).tolist()) # mat格式不支持set
        for label in label_set:
            if label != 1 and label != -1:
                raise ValueError('labels should be 1 or -1.')
                
    def calc_kernel(self, feats, kernel_option):
        '''计算核函数矩阵
        input:  feats(mat):训练样本的特征值
                kernel_option(tuple):核函数的类型以及参数
        output: kernel_matrix(mat):样本的核函数的值
        '''
        m = np.shape(feats)[0] # 样本的个数
        kernel_matrix = np.mat(np.zeros((m, m))) # 初始化样本之间的核函数值
        for i in range(m):
            kernel_matrix[:, i] = self.cal_kernel_value(feats, feats[i, :], kernel_option)
        return kernel_matrix
    
    def cal_kernel_value(self, feats, feats_i, kernel_option):
        '''样本之间的核函数的值
        input:  feats(mat):训练样本(n_sample, n_feat)
                feats_i(mat):第i个训练样本
                kernel_option(tuple):核函数的类型以及参数
        output: kernel_value(mat):样本之间的核函数的值
        '''
        kernel_type = kernel_option[0] # 核函数的类型，分为rbf和其他
        m = np.shape(feats)[0] # 样本的个数
        
        kernel_value = np.mat(np.zeros((m, 1)))
        
        if kernel_type == 'rbf': # rbf核函数
            sigma = kernel_option[1]
            if sigma == 0:
                sigma = 1.0
            for i in range(m):
                diff = feats[i, :] - feats_i   # (2,) - (2,) = (2,)
                kernel_value[i] = np.exp(diff * diff.T / (-2.0 * sigma**2))
        else: # 不使用核函数
            kernel_value = feats * feats_i.T
        return kernel_value
    
    def train(self):       
        entireSet = True
        alpha_pairs_changed = 0
        iteration = 0
        
        while (iteration < self.max_iter) and ((alpha_pairs_changed > 0) or entireSet): # 现在整个样本空间里循环
            print("\t iterration: ", iteration)
            alpha_pairs_changed = 0
    
            if entireSet:
                # 对所有的样本
                for x in range(self.n_samples):
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1
            else:
                # 非边界样本
                bound_samples = []
                for i in range(self.n_samples):
                    if self.alphas[i,0] > 0 and self.alphas[i,0] < self.C:
                        bound_samples.append(i)
                for x in bound_samples:
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1
            
            # 在所有样本和非边界样本之间交替
            if entireSet:
                entireSet = False
            elif alpha_pairs_changed == 0:
                entireSet = True
        
        self.trained = True
                
    def choose_and_update(self, alpha_i):
        '''判断和选择两个alpha进行更新
        input:  svm:SVM模型
                alpha_i(int):选择出的第一个变量
        '''
        error_i = self.cal_error(alpha_i) # 计算第一个样本的E_i
        
        # 判断选择出的第一个变量是否违反了KKT条件
        if (self.labels[alpha_i] * error_i < -self.toler) and (self.alphas[alpha_i] < self.C) or\
            (self.labels[alpha_i] * error_i > self.toler) and (self.alphas[alpha_i] > 0):
    
            # 1、选择第二个变量
            alpha_j, error_j = self.select_second_sample_j(alpha_i, error_i)
            alpha_i_old = self.alphas[alpha_i].copy()
            alpha_j_old = self.alphas[alpha_j].copy()
    
            # 2、计算上下界
            if self.labels[alpha_i] != self.labels[alpha_j]:
                L = max(0, self.alphas[alpha_j] - self.alphas[alpha_i])
                H = min(self.C, self.C + self.alphas[alpha_j] - self.alphas[alpha_i])
            else:
                L = max(0, self.alphas[alpha_j] + self.alphas[alpha_i] - self.C)
                H = min(self.C, self.alphas[alpha_j] + self.alphas[alpha_i])
            if L == H:
                return 0
    
            # 3、计算eta
            eta = 2.0 * self.kernel_mat[alpha_i, alpha_j] - self.kernel_mat[alpha_i, alpha_i] \
                      - self.kernel_mat[alpha_j, alpha_j]
            if eta >= 0:
                return 0
    
            # 4、更新alpha_j
            self.alphas[alpha_j] -= self.labels[alpha_j] * (error_i - error_j) / eta
    
            # 5、确定最终的alpha_j
            if self.alphas[alpha_j] > H:
                self.alphas[alpha_j] = H
            if self.alphas[alpha_j] < L:
                self.alphas[alpha_j] = L
    
            # 6、判断是否结束      
            if abs(alpha_j_old - self.alphas[alpha_j]) < 0.00001:
                self.update_error_tmp(alpha_j)
                return 0
    
            # 7、更新alpha_i
            self.alphas[alpha_i] += self.labels[alpha_i] * self.labels[alpha_j] \
                                    * (alpha_j_old - self.alphas[alpha_j])
    
            # 8、更新b
            b1 = self.b - error_i - self.labels[alpha_i] * (self.alphas[alpha_i] - alpha_i_old) \
                                                        * self.kernel_mat[alpha_i, alpha_i] \
                                 - self.labels[alpha_j] * (self.alphas[alpha_j] - alpha_j_old) \
                                                        * self.kernel_mat[alpha_i, alpha_j]
            b2 = self.b - error_j - self.labels[alpha_i] * (self.alphas[alpha_i] - alpha_i_old) \
                                                        * self.kernel_mat[alpha_i, alpha_j] \
                                 - self.labels[alpha_j] * (self.alphas[alpha_j] - alpha_j_old) \
                                                        * self.kernel_mat[alpha_j, alpha_j]
            if (0 < self.alphas[alpha_i]) and (self.alphas[alpha_i] < self.C):
                self.b = b1
            elif (0 < self.alphas[alpha_j]) and (self.alphas[alpha_j] < self.C):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
    
            # 9、更新error
            self.update_error_tmp(alpha_j)
            self.update_error_tmp(alpha_i)
    
            return 1
        else:
            return 0
    
    def cal_error(self, alpha_k):
        '''误差值的计算
        input:  svm:SVM模型
                alpha_k(int):选择出的变量
        output: error_k(float):误差值
        '''
        output_k = float(np.multiply(self.alphas, self.labels).T * self.kernel_mat[:, alpha_k] + self.b)
        error_k = output_k - float(self.labels[alpha_k])
        return error_k
    
    def update_error_tmp(self, alpha_k):
        '''重新计算误差值
        input:  svm:SVM模型
                alpha_k(int):选择出的变量
        output: 对应误差值
        '''
        error = self.cal_error(alpha_k)
        self.error_tmp[alpha_k] = [1, error]
    
    def select_second_sample_j(self, alpha_i, error_i):
        '''选择第二个样本
        input:  svm:SVM模型
                alpha_i(int):选择出的第一个变量
                error_i(float):E_i
        output: alpha_j(int):选择出的第二个变量
                error_j(float):E_j
        '''
        # 标记为已被优化
        self.error_tmp[alpha_i] = [1, error_i]
        candidateAlphaList = np.nonzero(self.error_tmp[:, 0].A)[0]
        
        maxStep = 0
        alpha_j = 0
        error_j = 0
    
        if len(candidateAlphaList) > 1:
            for alpha_k in candidateAlphaList:
                if alpha_k == alpha_i: 
                    continue
                error_k = self.cal_error(alpha_k)
                if abs(error_k - error_i) > maxStep:
                    maxStep = abs(error_k - error_i)
                    alpha_j = alpha_k
                    error_j = error_k
        else: # 随机选择          
            alpha_j = alpha_i
            while alpha_j == alpha_i:
                alpha_j = int(np.random.uniform(0, self.n_samples))
            error_j = self.cal_error(alpha_j)
        
        return alpha_j, error_j
    
    def predict_single(self, test_sample_x):
        '''利用SVM模型对每一个样本进行预测：预测流程需要，训练数据feat/label/alphas/b
        预测流程是：先计算测试样本跟训练样本的核函数输出K(xi, x0), 然后计算yi*alphi_i*K +b
        input:  
            svm:SVM模型
            test_sample_x(mat):样本
        output: 
            predict(float):对样本的预测
        '''
        test_sample_x = np.mat(test_sample_x)
        # 1、计算核函数矩阵
        kernel_value = self.cal_kernel_value(self.feats, test_sample_x, self.kernel_opt)
        # 2、计算预测值
        predict = kernel_value.T * np.multiply(self.labels, self.alphas) + self.b
        return np.sign(predict)
        
    
    def cal_accuracy(self, test_x, test_y):
        '''计算预测的准确性
        input:  svm:SVM模型
                test_x(mat):测试的特征
                test_y(mat):测试的标签
        output: accuracy(float):预测的准确性
        '''
        assert isinstance(test_x, np.matrix) and isinstance(test_y, np.matrix), 'feats and labels should be mat.'
        assert test_y.ndim ==2 and test_y.shape[1] == 1, 'labels should be (n,1) style mat.'
        
        # assert labels are [-1, 1]
        label_set = set(np.array(test_y).flatten().astype(np.int8).tolist()) # mat格式不支持set
        for label in label_set:
            if label != 1 and label != -1:
                raise ValueError('labels should be 1 or -1.')
        
        n_samples = np.shape(test_x)[0] # 样本的个数
        correct = 0.0
        for i in range(n_samples):
            # 对每一个样本得到预测值
            predict=self.predict_single(test_x[i, :])
            # 判断每一个样本的预测值与真实值是否一致
            if predict == test_y[i]:
                correct += 1
        accuracy = correct / n_samples
        return accuracy

    