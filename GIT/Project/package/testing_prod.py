#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:09:52 2020

@author: arindam
"""

from package.config import config as cfg
from package.preprocessor.data_management import load_training_data
from package.preprocessor.data_management import load_pipelines
from package.preprocessor.data_management import load_model
from package.preprocessor.data_management import store_test_accuracy
from package.preprocessor.data_management import store_test_confusion_matrix

import package.pipelines as pipe

df_data = load_training_data()
testing_data = pipe.get_test_data.fit_transform(df_data)
testing_label = pipe.get_test_label_data.fit_transform(df_data)
#tfidf_data = pipe.get_term_frequency.fit_transform(training_data)
# save the pipeline for test case use
#save_pipelines(pipe.get_term_frequency)
tfidf_load = load_pipelines(cfg.SAVED_PIPELINES_PATH)
tfidf_data = tfidf_load.transform(testing_data)

model=load_model(cfg.SAVED_MODEL_PATH)
test_predict = model.predict(tfidf_data)
## save model for later use 
#save_model(model)

#y_predicted = model.predict(tfidf_data)
accuracy_score = store_test_accuracy(testing_label,test_predict)
conf_matrix = store_test_confusion_matrix(testing_label,test_predict)
