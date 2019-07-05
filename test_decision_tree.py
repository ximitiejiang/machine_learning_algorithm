#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:14:32 2019

@author: ubuntu
"""

import numpy as np
from dataset.loan_dataset import LoanDataset
from dataset.iris_dataset import IrisDataset
from dataset.nonlinear_dataset import NonlinearDataset
from core.decision_tree_lib import CARTClf, ID3Clf, C45Clf, CARTReg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

def standardize(X):
    """ Standardize the dataset X """
    X_std = X
    mean = X.mean(axis=0)    # 按列求均值
    std = X.std(axis=0)      # 按列求标准差
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]  # 每一列特征单独做自己的标准化(减列均值，除列标准差)
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std

if __name__ == "__main__":
    
    source = 'iris'
    
    if source == 'treedata':  # 2 classes: from book of zhaozhiyong
        data = []
        with open('./dataset/simple/treedata.txt') as f:
            for line in f.readlines():
                sample = []
                lines = line.strip().split("\t")
                for x in lines:
                    sample.append(float(x))  # 转换成float格式
                data.append(sample)
        data = np.array(data)        # (200, 2)
        idx = np.arange(len(data))
        idx = np.random.permutation(idx)
        data = data[idx]
        
        x = data[:, :-1]  # (400,2)
        y = data[:, -1]  # (400,)
        cart = CARTClf(x, y)
        cart.evaluation(x,y)
        cart.vis_boundary(plot_step=0.01)
    
    if source == 'loan': # from lihang
        dataset = LoanDataset()
        x = dataset.datas
        y = dataset.labels
#        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        
        cart = CARTClf(x, y)
        cart.evaluation(x, y)
        print("final tree depth: %d, final gini: %d"%(cart.tree_final_params['final_depth'],
                                                      cart.tree_final_params['final_gini']))
    
    if source == 'moon':
        dataset = NonlinearDataset(type= 'circle', n_samples=300, noise=0.05, 
                                   label_transform_dict={1:1, 0:-1})
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3) # (n, 13) (n,)
        # array to mat
#        train_x, test_x, train_y, test_y = np.mat(train_x), np.mat(test_x), np.mat(train_y), np.mat(test_y)
        
        cart = CARTClf(train_x, train_y, min_samples_split=2)
        acc1 = cart.evaluation(train_x, train_y)
        print('train acc = %f'%(acc1))
        
        acc2 = cart.evaluation(test_x, test_y)
        print('test acc = %f'%(acc2))
        
        cart.vis_boundary(plot_step=0.05)

        
    if source == 'iris':
        dataset = IrisDataset()
        x = dataset.datas
        y = dataset.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        cart = CARTClf(train_x, train_y)
        cart.evaluation(train_x, train_y)
        cart.evaluation(test_x, test_y)
        

    if source == 'reg':
        data = pd.read_csv('./dataset/simple/TempLinkoping2016.txt', sep="\t")

        time = np.atleast_2d(data["time"].values).T
        temp = np.atleast_2d(data["temp"].values).T
    
        X = standardize(time)        # Time. Fraction of the year [0, 1]
        y = temp[:, 0]  # Temperature. Reduce to one-dim
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
        model = CARTReg(X, y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        y_pred_line = model.predict(X)
    
        # Color map
        cmap = plt.get_cmap('viridis')
    
#        mse = mean_squared_error(y_test, y_pred)
    
#        print ("Mean Squared Error:", mse)
    
        # Plot the results
        # Plot the results
    #    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
        m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    #    m3 = plt.scatter(366 * X_test, y_pred, color='black', s=10)
    #    m3 = plt.plot(366 * X_test, y_pred, color='black')
        train_x_sorted = np.sort(X_train, axis=0)
        train_y_sorted = np.array(y_train)[np.argsort(X_train, axis=0)]
        m1 = plt.plot(366 * train_x_sorted, train_y_sorted, color='red', linestyle='--')
        
        x_sorted = np.sort(X_test, axis=0)
        y_sorted = np.array(y_pred)[np.argsort(X_test, axis=0)]
        m3 = plt.plot(366 * x_sorted, y_sorted, color='blue')
        
        plt.suptitle("Regression Tree")
#        plt.title("MSE: %.2f" % mse, fontsize=10)
        plt.xlabel('Day')
        plt.ylabel('Temperature in Celcius')
        plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
        plt.show()
            



