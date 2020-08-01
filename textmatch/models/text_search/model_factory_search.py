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
from utils.logging import log_init
logging = log_init(const.LOG_PATH)
from textmatch.models.text_search.bm25 import BM25
from textmatch.config.constant import Constant as const
from textmatch.models.text_search.edit_sim import EditDistance
from textmatch.models.text_search.jaccard_sim import Jaccard


'''
'''

class ModelFactorySearch(object):
    '''match model factory
    '''
    def __init__(self, match_models=['bm25', 'edit_sim', 'jaccard_sim'],
                       bm25_model = BM25,
                       edit_sim_model=EditDistance,
                       jaccard_sim_model=Jaccard
                       ):

       self.model = {}
       for match_model in match_models:
           if match_model == 'bm25':
               model = bm25_model()
               self.model[match_model] = model
           elif match_model == 'edit_sim':
               model = edit_sim_model()
               self.model[match_model] = model
           elif match_model == 'jaccard_sim':
               model = jaccard_sim_model()
               self.model[match_model] = model
           else:
               logging.error( "[ModelFactory] match_model not existed，please select from ['bm25', 'edit_sim', 'jaccard_sim' " )
               continue
    def init(self, words_dict=None, update=False):
        if words_dict != None:
            self.id_lists, self.words_list = self._dic2list(words_dict)
        else:
            self.id_lists, self.words_list = None, None
        for key, model in self.model.items():
           self.model[key] = model.init(self.words_list)


    # id list / words list
    def _dic2list(self, words_dict):
       return list( words_dict.keys() ) , list( words_dict.values() )


    def predict(self, words):
        pre_dict = {}
        for key, model in self.model.items():
            pre_list = []
            pre = model.predict(words)
            for words_id, socre in zip(self.id_lists, pre):
                pre_list.append( (words_id, socre) )
            pre_dict[key] = pre_list
        return pre_dict
