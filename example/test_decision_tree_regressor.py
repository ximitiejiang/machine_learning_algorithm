#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 08:46:13 2019

@author: ubuntu
"""
import numpy as np
from dataset.loan_dataset import LoanDataset
from dataset.iris_dataset import IrisDataset
from dataset.nonlinear_dataset import NonlinearDataset
from dataset.regression_dataset import RegressionDataset
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
    
    source = 'linear'
    
    if source == 'temp':
        data = pd.read_csv('../dataset/simple/TempLinkoping2016.txt', sep="\t")

        time = np.atleast_2d(data["time"].values).T
        temp = np.atleast_2d(data["temp"].values).T
    
        X = standardize(time)        # Time. Fraction of the year [0, 1]
        y = temp[:, 0]  # Temperature. Reduce to one-dim
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
        model = CARTReg(X_train, y_train).train()
        y_pred = model.evaluation(X_test, y_test, show=True)
        
    if source == 'linear':
        dataset = RegressionDataset(n_samples=500, n_features=1, n_targets=1, noise=3)
        X = dataset.datas
        y = dataset.labels
#        plt.scatter(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        plt.scatter(X_train, y_train, label='train', color='red')
        plt.scatter(X_test, y_test, label='test', color='blue')
        plt.legend(loc='best')
        plt.grid()
        
        model = CARTReg(X_train, y_train).train()
        y_pred = model.evaluation(X_test, y_test, show=True)
           