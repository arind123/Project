#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:32:02 2020

@author: arindam
"""

from package.config import config as cfg
from package.preprocessor.data_management import load_training_data
from package.preprocessor.data_management import save_pipelines
from package.preprocessor.data_management import save_model
from package.preprocessor.data_management import store_accuracy
from package.preprocessor.data_management import store_confusion_matrix
import package.pipelines as pipe
from sklearn.linear_model import PassiveAggressiveClassifier as pac




df_data = load_training_data()
training_data = pipe.get_train_data.fit_transform(df_data)
training_label = pipe.get_train_label_data.fit_transform(df_data)
tfidf_data = pipe.get_term_frequency.fit_transform(training_data)
# save the pipeline for test case use
save_pipelines(pipe.get_term_frequency)

model=pac(max_iter=cfg.PAC_MAX_ITER)
model.fit(tfidf_data,training_label)
## save model for later use 
save_model(model)

y_predicted = model.predict(tfidf_data)
accuracy_score = store_accuracy(training_label,y_predicted)
conf_matrix = store_confusion_matrix(training_label,y_predicted)
