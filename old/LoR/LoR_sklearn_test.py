#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:33:23 2018

@author: suliang
"""

def fit(C, penalty, tol):
    from sklearn.linear_model import LogisticRegression
    
    LR = LogisticRegression(C=10, penalty='l1', tol=0.01)
    
    '''
    C : float, default: 1.0  
    Inverse of regularization strength; 
    must be a positive float. Like in support vector machines, 
    smaller values specify stronger regularization.
    代表正则项，越小值则越强的正则化
    
    penalty : str, ‘l1’ or ‘l2’, default: ‘l2’ 
    Used to specify the norm used in the penalization. 
    The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
    代表正则方式，l1正则活着l2正则，牛顿解法等只支持l2正则
    
    tol : float, default: 1e-4  
    Tolerance for stopping criteria.
    代表误差，停止拟合的最大误差
    
    dual : bool, default: False 
    Dual or primal formulation. Dual formulation is only implemented 
    for l2 penalty with liblinear solver. 
    Prefer dual=False when n_samples > n_features.
    对偶，一般在样品数较少时使用，如果样品数多于特征数不建议用。且只能配合l2正则和线性求解法用
    
    solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},
    default: ‘liblinear’ Algorithm to use in the optimization problem.
    For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and
    ‘saga’ are faster for large ones.
    For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
    handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
    ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas
    ‘liblinear’ and ‘saga’ handle L1 penalty.
    求解器，默认线性求解对小数据集比较合适，而sag/saga会相对快一点
    线性求解和saga求解可以配合l1正则
    多分类问题，必须选择牛顿-cg， sag, saga, lbfgs求解器，且只能选择l2正则。
    
    '''
    
    LR.fit(X, y)

