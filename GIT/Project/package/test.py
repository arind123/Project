#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:47:01 2020

@author: arindam
"""

def get_label(df):
        X = df.copy()
        labels = X['label']
        return labels
    
A = get_label()
