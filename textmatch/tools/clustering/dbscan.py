# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  DBSCAN
   Author :       machinelp
   Date :         2020-06-15
-------------------------------------------------

'''

import numpy as np
from sklearn.cluster import DBSCAN


class DBSCANClustering(object):
    def __init__(self, eps=0.5, min_samples=5):
        self.db_scan = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_(self, data_list):
        self.db_scan.fit(data_list)
        return self
    
    def predict_(self, data_list):
        label_list = self.db_scan.predict(data_list)
        return label_list

    def predict(self, data_list):
        label_list = self.db_scan.fit_predict(data_list)
        return label_list
    
    def save_model_(self, save_path):
        #保存模型
        joblib.dump(self.db_scan, save_path)

