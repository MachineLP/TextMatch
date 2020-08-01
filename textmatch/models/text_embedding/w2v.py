# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  w2v
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import time
import jieba
import gensim
import threading
import numpy as np
from .stop_words import StopWords
from textmatch.config.config import cfg
from textmatch.utils.logging import logging
from textmatch.config.constant import Constant as const
from textmatch.models.model_base.model_base import ModelBase


class Word2VecBase():
    '''
    '''
    _instance_lock = threading.Lock()
    
    def __init__(self, 
                      w2v_model_file=const.W2V_MODEL_FILE, 
                      stop_word=StopWords(stopwords_file=const.STOPWORDS_FILE)  ):
       self.w2v_model = gensim.models.Word2Vec.load(w2v_model_file)
       self.stop_word = stop_word
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with Word2VecBase._instance_lock:
                if not hasattr(cls, '_instance'):
                    Word2VecBase._instance = super().__new__(cls)

            return Word2VecBase._instance
    # 
    def word2vec_inference(self, word):
       return self.w2v_model[word] 
   
    def _predict(self, words, del_stopword=False):
       word_list = jieba.cut(words,cut_all=False)
       if del_stopword:
          word_list = self.stop_word.del_stopwords(word_list)
       zero_vec = np.zeros(256)
       word_vector_list = []
       for word in word_list: 
           try:
               word_vector_list.append( self.w2v_model[word] )
           except:
               word_vector_list.append(zero_vec) 
       word_vector_list = np.array(word_vector_list).mean(axis=0)
       return word_vector_list[np.newaxis, :].astype(float)


class Word2Vec(ModelBase):
    '''
    '''
    def __init__(self, 
                      w2v_model_file=const.W2V_MODEL_FILE, 
                      stop_word=StopWords(stopwords_file=const.STOPWORDS_FILE)  ):
       self.w2v_model = Word2VecBase( w2v_model_file,  stop_word)
       self._predict = self.w2v_model._predict
   

    def init(self, words_list=None, update=True, del_stopword=cfg.emb.DEL_STOPWORD): 
       if words_list!=None:
           self.words_list_pre = [] 
           for words in words_list:
              self.words_list_pre.append( self.w2v_model._predict(words, del_stopword)[0] )
           self.words_list_pre = self._normalize(self.words_list_pre)
       return self
    
    def predict(self, words, del_stopword=cfg.emb.DEL_STOPWORD):
       pre = [self.w2v_model._predict(words, del_stopword)[0]]
       pre = self._normalize(pre)
       return np.dot( self.words_list_pre[:], pre[0] ) 

