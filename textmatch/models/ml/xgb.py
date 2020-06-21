# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  XGB
   Author :       machinelp
   Date :         2020-06-13
-------------------------------------------------

'''
import sys
import logging
import numpy as np
from textmatch.config.config import cfg
from textmatch.config.constant import Constant as const
import xgboost as xgb


class XGB:

    def __init__(self):
        self.other_params = {}
        for k, v in cfg.xgb.items():
            print ('XGB params:',k,'>>>>',v)
            self.other_params[k] = v
        #self.other_params = {'learning_rate': cfg.xgb.learning_rate,
        #                     'max_depth':cfg.xgb.max_depth,
        #                     'n_estimators':cfg.xgb.n_estimators
        #                     }
        self.clf = xgb.XGBClassifier(**self.other_params)
        pass

    def fit(self, X_train, y_train ):
        self.clf.fit(X_train, y_train)  
        pass

    def predict(self, X_test):
        predict = self.clf.predict_proba(X_test)[:,1]
        return predict

    def save_model(self):
        pass

    def load_model(self):
        pass
















