#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 18:33:48 2019

@author: ubuntu
"""

import numpy as np
a = np.ones((256,1,10,10))

k = np.zeros((9,1))
i = np.ones((9,64))
j = np.ones((9,64))

a[:, k, i, j]
