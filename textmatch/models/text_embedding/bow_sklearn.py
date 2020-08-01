# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  tf-idf-sklearn 
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import os
import jieba
import pickle
import logging
import numpy as np
from .stop_words import StopWords
from textmatch.config.config import cfg
from textmatch.utils.logging import log_init
logging = log_init(const.LOG_PATH)
from textmatch.config.constant import Constant as const
from textmatch.models.model_base.model_base import ModelBase
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class Bow(ModelBase):

    def __init__( self, 
                        dic_path=const.BOW_DIC_PATH, 
                        bow_index_path=const.BOW_INDEX_PARH,
                        stop_word=StopWords ):
        '''
        '''

        self.dic_path = dic_path
        self.bow_index_path = bow_index_path
        for per_path in [self.dic_path, self.bow_index_path]:
            per_path = '/'.join(per_path.split('/')[:-1])
            if os.path.exists(per_path) == False:
                os.makedirs(per_path)
        self.stop_word = stop_word() 
        self.vectorizer = CountVectorizer(stop_words = None, max_df=cfg.emb.MAX_DF, min_df=cfg.emb.MIN_DF, max_features=cfg.emb.MAX_FEATURES, token_pattern='(?u)\\b\\w\\w*\\b')  

    # init
    def init(self, words_list=None, update=True):
        if (~os.path.exists(self.dic_path) or ~os.path.exists(self.bow_model_path) or ~os.path.exists(self.bow_index_path) or update) and (words_list!=None):
            word_list = self._seg_word(words_list)
        
        if os.path.exists(self.dic_path) and update==False:
            with open(self.dic_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            try:
                 logging.info('[Bow] start build tfidf model.')
                 if words_list==None:
                     logging.error( '[Bow] words_list is None' )
                 self._gen_model(word_list)
                 logging.info('[Bow] build tfidf model success.')
            except Exception as e:
                 logging.error( '[Bow] build tfidf model error，error info: {} '.format(e) )
        if words_list!=None:
            self.words_list_pre = []
            for per_word in words_list:
                self.words_list_pre.append( self._normalize( self._predict(per_word) )[0] )
            self.words_list_pre = np.array(self.words_list_pre)
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
    def _seg_word(self, words_list, jieba_flag=cfg.emb.JIEBA_FLAG, del_stopword=cfg.emb.DEL_STOPWORD):
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
    

    def fit(self, word_list):
        word_list = self._seg_word(word_list)
        self._gen_model(word_list)
    
    # build dic
    def _gen_dic(self, word_list):
        #print ('word_list>>>>>', word_list)
        dic = self.vectorizer.fit_transform(word_list)
        with open(self.dic_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        #print( 'vectorizer>>>>', self.vectorizer.get_feature_names() )
        return dic

    # build tf-idf model
    def _gen_model(self, word_list):
        self._gen_dic(word_list)
    
    def _predict(self, words):
        tf_idf_embedding = self.vectorizer.transform(self._seg_word([words]))
        tf_idf_embedding = tf_idf_embedding.toarray().sum(axis=0)
        # print ('>>>>', tf_idf_embedding[np.newaxis, :]) 
        return tf_idf_embedding[np.newaxis, :].astype(float)
    
    def predict(self, words):
        pre = self._normalize( self._predict(words) )
        return np.dot( self.words_list_pre[:], pre[0] ) 

