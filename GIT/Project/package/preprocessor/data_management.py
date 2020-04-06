#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:03:34 2020

@author: arindam
"""
import pandas as pd
from pickle import dump
from pickle import load
from package.config import config as cfg
from sklearn.metrics import accuracy_score

def load_training_data()-> pd.DataFrame():
    ## Load data 
    filepath = cfg.DATA_PATH
    df = pd.read_csv(filepath)
    return df

def save_pipelines(tfidf):
    ## Save Tfidf for transformin test set 
    filepath = cfg.SAVED_PIPELINES_PATH
    dump(tfidf, open(filepath, 'wb'))
    return

def load_pipelines(filepath):
    ## Save Tfidf for transformin test set 
    #filepath = cfg.SAVED_PIPELINES_PATH
    pipeline_load = load(open(filepath, 'rb'))
    return  pipeline_load

def save_model(model):
    ## Save Tfidf for transformin test set 
    filepath = cfg.SAVED_MODEL_PATH
    dump(model, open(filepath, 'wb'))
    return 

def load_model(filepath):
    ## Save Tfidf for transformin test set 
    #filepath = cfg.SAVED_MODEL_PATH
    model_load = load(open(filepath, 'rb'))
    return model_load

def store_accuracy(y_actual, y_predicted):
    ## Save Tfidf for transformin test set 
    score = accuracy_score(y_actual,y_predicted)
    output_file = open(cfg.TRAIN_RESULT_ACCURACY_PATH,'w')
    output_file.write('Accuracy: %1.2f\n' % score)
    return score

def store_confusion_matrix(y_actual, y_predicted):
    ## Save Tfidf for transformin test set 
    y_actual = pd.Series(y_actual, name="Actual")
    y_predicted = pd.Series(y_predicted, name="Predicted")
    df_confusion = pd.crosstab(y_actual, y_predicted)
    df_confusion = pd.DataFrame(df_confusion)
    df_confusion.to_csv(cfg.TRAIN_RESULT_CM_PATH)
    return df_confusion


def store_test_accuracy(y_actual, y_predicted):
    ## Save Tfidf for transformin test set 
    score = accuracy_score(y_actual,y_predicted)
    output_file = open(cfg.TEST_RESULT_ACCURACY_PATH,'w')
    output_file.write('Accuracy: %1.2f\n' % score)
    return score

def store_test_confusion_matrix(y_actual, y_predicted):
    ## Save Tfidf for transformin test set 
    y_actual = pd.Series(y_actual, name="Actual")
    y_predicted = pd.Series(y_predicted, name="Predicted")
    df_confusion = pd.crosstab(y_actual, y_predicted)
    df_confusion = pd.DataFrame(df_confusion)
    df_confusion.to_csv(cfg.TEST_RESULT_CM_PATH)
    return df_confusion