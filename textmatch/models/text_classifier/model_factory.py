# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  Modelfactory
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import sys
import logging
import numpy as np
from textmatch.config.constant import Constant as const
from textmatch.models.text_classifier.dnn import DNN 

'''
'''

class ModelFactory(object):
    '''match model factory
    '''
    def __init__(self, match_models=['dnn', 'cnn', 'rnn'],
                       bow_model = DNN,
                       ):
       self.model = {}
       for match_model in match_models:
           if match_model == 'dnn':

           else:
               logging.error( "[text classifer ModelFactory] match_model not existed，please select from ['dnn', 'rnn', 'cnn', ...] " )
               continue

   
     
    



