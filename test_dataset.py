#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:44:08 2019

@author: ubuntu
"""
from dataset.boston_house_price_dataset import BostonHousePriceDataset
from dataset.breast_cancer_dataset import BreastCancerDataset
from dataset.digits_dataset import DigitsDataset
from dataset.iris_dataset import IrisDataset
from dataset.mnist_dataset import MnistDataset
from dataset.multi_class_dataset import MultiClassDataset
from dataset.nonlinear_dataset import NonlinearDataset
from dataset.olivetti_faces_dataset import OlivettiFacesDataset
from dataset.lfw_faces_dataset import LFWPeopleDataset, LFWPairsDataset
from dataset.heart_scale_dataset import HeartScaleDataset
from dataset.diabetes_dataset import DiabetesDataset
from dataset.pima_indians_diabetes_dataset import PimaIndiansDiabetesDataset
from dataset.loan_dataset import LoanDataset
from dataset.regression_dataset import RegressionDataset

# 1
#bost = BostonHousePriceDataset()
#data, label = bost[10]

## 2
#bcset = BreastCancerDataset()
#data, label = bcset[10]                     
#bcset.statistics()      

# 3
#dg = DigitsDataset()
#dg.statistics()
#idx = 223
#dg.show(idx)
#img, label = dg[idx]
#print('label = %d' % label)

# 4
#iris = IrisDataset()
#data, label = iris[10]                     
#iris.statistics()     

# 5
#idx = 5
#mn = MnistDataset(root_path='./dataset/mnist/')
#mn.statistics()
#data, label = mn[idx]
#print('label is: %d'%label)
#mn.show(idx)

# 6
#mc = MultiClassDataset(100, 4, 2)
#data, label = mc[10]                     
#mc.statistics()  

# 7
#nl = NonlinearDataset('moon', 100, 0.05)
#data, label = nl[10]                     
#nl.statistics()  

# 8
#faces = OlivettiFacesDataset()
#faces.statistics()
#idx = 65
#data, label = faces[idx]
#faces.show(idx)     
#print('this pic idx = %d, label = %d' % (idx, label))

# 9
#faces = LFWPeopleDataset(min_faces_per_person=2)
#faces.statistics()
#idx = 65
#data, label = faces[idx]
#faces.show(idx)     
#print('this pic idx = %d, label = %d' % (idx, label))

# 10
#pair = LFWPairsDataset()
#pair.statistics()
#idx =1268
#data, label = pair[idx]
#pair.show(idx)
#print('this pic idx = %d, label = %d' % (idx, label))

# 11
#hs = HeartScaleDataset()
#hs.statistics()

# 12
#db = DiabetesDataset()
#db.statistics()

#13
#db = PimaIndiansDiabetesDataset()
#idx = 15
#data, label = db[idx]
#print('data is: ', data)
#print('label is: %d'%label)

# 14
#loan = LoanDataset()
#idx = 10
#data, label = loan[idx]
#print('data is: ', data)
#print('label is: %d'%label)

# 15
reg = RegressionDataset(n_features=2, n_samples=100, noise=1)
datas = reg.datas
labels = reg.labels
