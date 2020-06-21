# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  MatchModelBase
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''
import numpy as np

class ModelBase(object):
    '''
    '''
    def __init__(self, ):
       pass
   

    def _init(self, ):
       pass
    
    def _normalize(self, x):
       x /= (np.array(x)**2 + 0.00000001).sum(axis=1, keepdims=True)**0.5 
       return x

    def predict(self, words):
       pass

