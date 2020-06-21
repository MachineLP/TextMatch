# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  PCA
   Author :       machinelp
   Date :         2020-06-15
-------------------------------------------------

'''

import numpy as np
from sklearn.externals import joblib 
from sklearn.decomposition import PCA


class PCADecomposition(object):
    def __init__(self, n_components=256):
        self.pca = PCA(n_components=n_components)

    def fit(self, data):
        self.pca.fit( data )
        return self
    
    def transform(self, data):
        return self.pca.transform( data )
    
    def save_model(self, save_path):
        #保存模型
        joblib.dump(self.pca, save_path)

