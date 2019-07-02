#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:07:14 2019

@author: suliang
"""

class GridSearch():
    
    def __init__(self, model, params_dict):
        """
        Args:
            params_dict:  为数据的参数字典，比如{'C':[0.1, 1, 10, 100], 'sigma': [0.01, 0.1, 0.5, 1]}
        """
        self.model = model
        self.params_dict = params_dict
        
    def search(self):
        for key, value in self.params_dict.items():
            param_group.append()
        
        
        return best_params_dict
    
    def show_result(self):
        pass