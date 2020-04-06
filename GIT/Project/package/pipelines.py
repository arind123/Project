#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:51:37 2020

@author: arindam
"""


from sklearn.pipeline import Pipeline
from package.preprocessor import pre_processor as pre_prop
from package.config import config as cfg
from sklearn.feature_extraction.text import TfidfVectorizer as tf



    
get_train_data = Pipeline([
                        ('get_train_data',  pre_prop.GetTraindata()),
                        #('term_frequency',tf(stop_words= cfg.STOP_WORDS, max_df = cfg.MAX_DF)),
                        #('pa_classifier', pa(max_iter = cfg.PA_MAX_ITER))
                     ])

get_train_label_data = Pipeline([
                        ('get_train_label_data',  pre_prop.GetTrainlabel()),
                        #('term_frequency',tf(stop_words= cfg.STOP_WORDS, max_df = cfg.MAX_DF)),
                        #('pa_classifier', pa(max_iter = cfg.PA_MAX_ITER))
                     ])
get_test_data = Pipeline([
                        ('get_test_data',  pre_prop.GetTestdata()),
                        #('term_frequency',tf(stop_words= cfg.STOP_WORDS, max_df = cfg.MAX_DF)),
                        #('pa_classifier', pa(max_iter = cfg.PA_MAX_ITER))
                     ])

get_test_label_data = Pipeline([
                        ('get_test_label_data',  pre_prop.GetTestlabel()),
                        #('term_frequency',tf(stop_words= cfg.STOP_WORDS, max_df = cfg.MAX_DF)),
                        #('pa_classifier', pa(max_iter = cfg.PA_MAX_ITER))
                     ])

get_term_frequency = Pipeline([
                        ('get_term_frequency',  tf(stop_words = cfg.STOP_WORDS, max_df = cfg.MAX_DF)),
                        #('term_frequency',tf(stop_words= cfg.STOP_WORDS, max_df = cfg.MAX_DF)),
                        #('pa_classifier', pa(max_iter = cfg.PA_MAX_ITER))
                     ])
