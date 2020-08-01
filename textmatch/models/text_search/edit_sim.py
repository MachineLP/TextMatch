# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  edit sim
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import os
import jieba
import pickle
import distance
import numpy as np
from textmatch.config.config import Config as conf
from textmatch.config.constant import Constant as const
from textmatch.models.model_base.model_base import ModelBase
from textmatch.models.text_embedding.stop_words import StopWords
from textmatch.utils.logging import log_init
logging = log_init(const.LOG_PATH)


class EditDistance(ModelBase):

    def __init__( self, stop_word=StopWords ):
        '''
        '''
        self.stop_word = stop_word() 

    # init
    def init(self, words_list=None, update=True):
        self.word_list = self._seg_word(words_list)
        return self

    '''
    # seg word
    def _seg_word(self, words_list, jieba_flag=True, del_stopword=False):
        if jieba_flag:
            word_list = [[self.stop_word.del_stopwords(words) if del_stopword else word for word in jieba.cut(words)] for words in words_list]
        else:
            word_list = [[self.stop_word.del_stopwords(words) if del_stopword else word for word in words] for words in words_list]
        print( 'word_list>>>', word_list )
        return [ ' '.join(word) for word in word_list  ]
    '''
    # seg word
    def _seg_word(self, words_list, jieba_flag=conf.JIEBA_FLAG, del_stopword=conf.DEL_STOPWORD):
        word_list = []
        if jieba_flag:
            for words in words_list:
                if del_stopword:
                    if words!='' and type(words) == str:
                        word_list.append( [word for word in self.stop_word.del_stopwords(jieba.cut(words))] )
                else:
                    if words!='' and type(words) == str:
                        word_list.append( [word for word in jieba.cut(words)] )
        else:
            for words in words_list:
                if del_stopword:
                    if words!='' and type(words) == str:
                        word_list.append( [word for word in self.stop_word.del_stopwords(words)] )
                else:
                    if words!='' and type(words) == str:
                        word_list.append( [word for word in words] )
        return [ ' '.join(word) for word in word_list  ]
    
    def edit_sim(self, s1, s2):
        maxLen = max(len(s1), len(s2))
        dis = distance.levenshtein(s1, s2)
        sim = 1 - dis * 1.0 / maxLen
        return sim

    def predict(self, words):
        word = self._seg_word([words])[0]
        results = [ self.edit_sim(x, word) for x in self.word_list ]
        return results

