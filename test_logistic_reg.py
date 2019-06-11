#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:44 2019

@author: ubuntu
"""

import matplotlib.pyplot as plt
from dataset.breast_cancer_dataset import BreastCancerDataset
from core.logistic_reg_lib import LogisticReg
from sklearn.model_selection import train_test_split

def accuracy(preds, labels):
    """preds(n,), labels(n)
    """
    return acc

if __name__ == "__main__":

    # get dataset
    dataset = BreastCancerDataset()
    train_feats, test_feats, train_labels, test_labels = train_test_split(dataset.datas, dataset.labels, test_size=0.2)
    
    # get model
    logi = LogisticReg(train_feats, train_labels)
    logi.train()
    print(logi.W)
    print(logi.trained)

    # test and show
    pred_labels, probs = logi.classify(test_feats)