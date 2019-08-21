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
from utils.vis import voc_colors, city_colors
from utils.transformer import label_transform, label_to_onehot, onehot_to_label

class BaseModel():
    """ 所有分类回归模型的基类
    """
    def __init__(self, feats, labels):
        assert feats.ndim ==2, 'the feats should be (n_samples, m_feats), each sample should be 1-dim flatten data.'
        self.feats = feats
        self.labels = labels.astype(np.int8)
        self.trained = False
        self.model_dict = {}
        
        if self.labels.ndim >= 2:  # 独热编码标签处理
            self.classes_list = list(range(self.labels.shape[1]))
        else:   # 一维标签处理
            self.classes_list = sorted(list(set(np.array(self.labels).reshape(-1).tolist())))  # classes_list为排序从小到大
        self.n_classes = len(self.classes_list)
        self.n_samples = self.feats.shape[0]
        self.n_feats = self.feats.shape[1]
    
    def train(self):
        """训练函数，需要实现"""
        raise NotImplementedError('the classify func is not implemented.')
    
    def predict_single(self, single_sample_feats):
        """单样本分类函数，需要实现"""
        raise NotImplementedError('the classify func is not implemented.')
    
    def evaluation(self, test_feats, test_labels):
        """评价整个验证数据集
        Args:
            test_feats: (n_sample, n_feat)
            test_labels: (n_sample,)
        """
        # 独热恢复原始
        if test_labels.ndim >= 2:
            test_labels = np.argmax(test_labels, axis=1)
        correct = 0
        total_sample = len(test_feats)
        start = time.time()
        for feat, label in zip(test_feats, test_labels):
            pred_label = self.predict_single(feat)
            if int(pred_label) == int(label):
                correct += 1
        acc = correct / total_sample
        print('======%s======'%self.model_dict['model_name'])
        print('Finished evaluation in %f seconds with accuracy = %f.'%((time.time() - start), acc))
        
        return acc
    
    def vis_loss(self, losses):
        """可视化损失"""
        assert losses is not None and len(losses) != 0, 'can not visualize losses because losses is empty.'
        if losses.ndim == 2:
            x = np.array(losses)[:,0]
            y = np.array(losses)[:,1]
        else:
            x = np.arange(len(losses))
            y = np.array(losses)
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('losses')
        plt.plot(x,y)
    
    def vis_points_line(self, feats, labels, W):
        """可视化二维点和分隔线(单组w)，可适用于模型是线性分割平面，比如logistic/perceptron
        """
        assert feats.shape[1] == 2, 'feats should be 2 dimention data with 1st. column of 1.'
        assert len(W) == 3, 'W should be 3 values list.'
        
        # 如果是多分类的独热编码，需要逆变换为普通编码0-k
        if labels.ndim > 1:
            ori_labels = np.zeros((len(labels), ))
            for i in range(len(labels)):
                label = labels[i]
                idx = np.where(label==1)[0].item()
                ori_labels[i] = idx
            labels = np.array(ori_labels)
            
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
            y[i] = (-W[0] - x[i]*W[1]) / W[2]
        plt.plot(x, y, c='r')
    
    def vis_boundary(self, plot_step=0.02):
        """可视化分隔边界，可适用于线性可分和非线性可分特征，比较普适
        """
        assert self.feats.shape[1] == 2, 'feats should be 2 dimention data.'
        assert self.feats.ndim == 2 # 只能可视化边界二维特征
        xmin, xmax = self.feats[:,0].min(), self.feats[:,0].max()
        ymin, ymax = self.feats[:,1].min(), self.feats[:,1].max()
        xx, yy = np.meshgrid(np.arange(xmin, xmax, plot_step),
                             np.arange(ymin, ymax, plot_step))
        xx_flatten = xx.flatten()
        yy_flatten = yy.flatten()
        
        z = []
        for i in range(len(xx_flatten)):
            point = np.array([xx_flatten[i], yy_flatten[i]]) 
            z.append(self.predict_single(point))    # 获得预测
        zz = np.array(z).reshape(xx.shape).astype(np.int8)
        # 绘制等高线颜色填充
        plt.figure()
        plt.subplot(1,1,1)
        plt.contourf(xx, yy, zz, cmap=plt.cm.Paired)
        # 标签变换：4种标签 (10,) (10,1)array, (10,1)mat, (10,4)
        labels = np.array(self.labels)
        labels = labels.reshape(-1,1) if labels.ndim==1 else labels  # 统一成(10,1), (10,4)
        if labels.shape[1] > 1:  # (10,1), (10,4)
            labels = onehot_to_label(labels)  # 如果是多列，则转换成独热编码
        labels = label_transform(labels.reshape(-1), label_transform_dict={1:1, -1:0, 0:0}).astype(np.int8)# 转换成0,1编码
        colors = city_colors(self.n_classes, norm=True)  # 返回array
        colors = colors[labels]       #获取每个label的颜色代码    

        plt.scatter(np.array(self.feats)[:,0], 
                    np.array(self.feats)[:,1], 
                    c = colors)
        if self.model_dict:
            model_name = self.model_dict['model_name']
        else:
            model_name = 'model'
        plt.title('predict boundary of ' + model_name)
        
    
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
        
        
        