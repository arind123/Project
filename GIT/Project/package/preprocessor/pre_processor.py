#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 02:39:43 2020

@author: arindam
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from package.config import config as cfg


class GetTraindata(BaseEstimator, TransformerMixin):
    #to get train data
    def __init__(self):
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series([])
        self.y_test = pd.Series([])
        self.text_col = cfg.TO_DIVIDE_COL
        self.label = cfg.LABEL
    
    
    def fit(self, X, y=None):
        return self  #  does nothing

    def transform(self, df, y=None):
        X = df.copy()
        #label = X.apply(self.get_label)
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(X[self.text_col],X[self.label], test_size=cfg.TEST_SIZE , random_state= cfg.RANDOM_STATE )
        df = self.x_train.copy() 
        return df  # where the actual feature extraction happens

class GetTrainlabel(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series([])
        self.y_test = pd.Series([])
        self.text_col = cfg.TO_DIVIDE_COL
        self.label = cfg.LABEL
        

    def fit(self, X, y=None):
        return self  #  does nothing

    def transform(self, df, y=None):
        X = df.copy()
        #abel = X.apply(self.get_label)
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(X[self.text_col],X[self.label], test_size=cfg.TEST_SIZE , random_state= cfg.RANDOM_STATE )
        y = self.y_train
        return y  # where the actual feature extraction happens



class GetTestdata(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series([])
        self.y_test = pd.Series([])
        self.text_col = cfg.TO_DIVIDE_COL
        self.label = cfg.LABEL
        
    
    def fit(self, X, y=None):
        return self  #  does nothing

    def transform(self, df, y=None):
        X = df.copy()
        #label = X.apply(self.get_label)
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(X[self.text_col],X[self.label], test_size=cfg.TEST_SIZE , random_state= cfg.RANDOM_STATE )
        df = self.x_test.copy()
        return df  # where the actual feature extraction happens


class GetTestlabel(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series([])
        self.y_test = pd.Series([])
        self.text_col = cfg.TO_DIVIDE_COL
        self.label = cfg.LABEL
        
    
    def fit(self, X, y=None):
        return self  #  does nothing

    def transform(self, df, y=None):
        X = df.copy()
        #label = X.apply(self.get_label)
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(X[self.text_col],X[self.label], test_size=cfg.TEST_SIZE , random_state= cfg.RANDOM_STATE )
        y = self.y_test
        return y  # where the actual feature extraction happens





    
