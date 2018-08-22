#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 09:27:59 2018

@author: suliang
"""
import DG_v1

df_train, df_test = DG_v1.importData()

x_train, y_train, x_test = DG_v1.featureEngineering(df_train, df_test)

results = DG_v1.modelsFit(x_train, y_train)


# MyMLCode

