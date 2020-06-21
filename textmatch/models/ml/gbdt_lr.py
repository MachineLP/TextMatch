# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  GBDTLR
   Author :       machinelp
   Date :         2020-06-13
-------------------------------------------------

'''
import sys
import logging
import numpy as np
from textmatch.config.config import cfg
from textmatch.config.constant import Constant as const
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

class GBDTLR:
    def __init__(self):
        self.gbdt_other_params = {}
        for k, v in cfg.gbdt.items():
            print ('GBDT params:',k,'>>>>',v)
            self.gbdt_other_params[k] = v
        self.lr_other_params = {}
        for k, v in cfg.lr.items():
            print ('LR params:',k,'>>>>',v)
            self.lr_other_params[k] = v
        self.clf_gbdt = GradientBoostingClassifier(**self.gbdt_other_params)
        self.clf_lr = LogisticRegression(**self.lr_other_params)
        self.enc = OneHotEncoder()
        pass

    def fit(self, train_x, train_y):
        self.clf_gbdt.fit(train_x, train_y)
        train_new_feature = self.clf_gbdt.apply(train_x)
        train_new_feature = train_new_feature.reshape(-1, self.gbdt_other_params['n_estimators'])
        self.enc.fit(train_new_feature)
        train_new_feature2 = np.array(self.enc.transform(train_new_feature).toarray())
        self.clf_lr.fit(train_new_feature2, train_y)

        return self

    def predict(self, X_test):
        test_new_feature = self.clf_gbdt.apply(X_test)
        test_new_feature = test_new_feature.reshape(-1, self.gbdt_other_params['n_estimators'])
        test_new_feature2 = np.array(self.enc.transform(test_new_feature).toarray())
        predict = self.clf_lr.predict_proba(test_new_feature2)[:,1]
        return predict

    def save_model(self):
        pass

    def load_model(self):
        pass
















