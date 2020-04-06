#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 02:59:19 2020

@author: arindam
"""

LABEL = 'label'
TO_DIVIDE_COL = 'text'
TEST_SIZE = 0.2
RANDOM_STATE = 7
STOP_WORDS = 'english'
MAX_DF = 0.7
PAC_MAX_ITER = 50

DATA_PATH = '/home/arindam/Fake News Detection Project/news.csv'
SAVED_PIPELINES_PATH = '/home/arindam/package/preprocessor/tfidf.pkl'
SAVED_MODEL_PATH = '/home/arindam/package/trained_model/finalized_model.pkl'
TRAIN_RESULT_ACCURACY_PATH = '/home/arindam/package/result/train_accuracy.txt'
TRAIN_RESULT_CM_PATH = '/home/arindam/package/result/train_confusion_matrix.csv'
TEST_RESULT_ACCURACY_PATH = '/home/arindam/package/result/test_accuracy.txt'
TEST_RESULT_CM_PATH = '/home/arindam/package/result/test_confusion_matrix.csv'


