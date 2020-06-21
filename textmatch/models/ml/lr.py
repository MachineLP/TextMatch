# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  LR
   Author :       machinelp
   Date :         2020-06-13
-------------------------------------------------

'''
import sys
import logging
import numpy as np
from textmatch.config.config import cfg
from sklearn.linear_model import LogisticRegression
from textmatch.config.constant import Constant as const

class LR:

    def __init__(self):
        self.other_params = {}
        for k, v in cfg.lr.items():
            print ('LR params:',k,'>>>>',v)
            self.other_params[k] = v
        self.clf = LogisticRegression(**self.other_params) 
        pass

    def fit(self, train_x, train_y):
        self.clf.fit(train_x, train_y)
        return self

    def predict(self, X_test):
        predict = self.clf.predict_proba(X_test)[:,1]
        return predict

    def save_model(self):
        pass

    def load_model(self):
        pass
















