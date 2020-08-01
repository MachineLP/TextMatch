# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  Modelfactory
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import sys
import numpy as np
from textmatch.utils.logging import log_init
logging = log_init(const.LOG_PATH)
from textmatch.config.constant import Constant as const
from textmatch.models.text_classifier.dnn import DNN 
from textmatch.models.ml.lr import LR
from textmatch.models.ml.gbdt import GBDT
from textmatch.models.ml.gbdtlr import GBDTLR
from textmatch.models.ml.xgb import XGB
from textmatch.models.ml.lgb import LGB

'''
'''

class ModelFactoryClassifier(object):
    '''match model factory
    '''
    def __init__(self, match_models=['gbdt']
                       ):
        self.model = {}
        for match_model in match_models:
            if match_model == 'lr':
                model = LR()
                model.load_model()
                self.model[match_model] = model
            elif match_model == 'gbdt':
                model = GBDT()
                model.load_model()
                self.model[match_model] = model
            elif match_model == 'gbdtlr':
                model = GBDTLR()
                model.load_model()
                self.model[match_model] = model
            elif match_model == 'xgb':
                model = XGB()
                model.load_model()
                self.model[match_model] = model
            elif match_model == 'lgb':
                model = LGB()
                model.load_model()
                self.model[match_model] = model
            else:
                logging.error( "[text classifer ModelFactory] match_model not existed，please select from ['lr', 'gbdt', 'gbdtlr', 'xgb', 'lgb' ...] " )
                continue
    def init(self):
        pass

    def predict(self, emb):
        pre_dict = {}
        for key, model in self.model.items():
            pre_list = []
            pre = model.predict(emb) 
            pre_dict[key] = pre
        return pre_dict 

