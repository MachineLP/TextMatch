# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  tf-idf-sklearn 
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import os
import re
import jieba
import pickle
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

class NgramTfIdf(ModelBase):

    def __init__( self, 
                        dic_path=const.NGRAM_TFIDF_DIC_PATH, 
                        tfidf_model_path=const.NGRAM_TFIDF_MODEL_PATH, 
                        tfidf_index_path=const.NGRAM_TFIDF_INDEX_PATH,
                        stop_word=StopWords ):
        '''
        '''

        self.dic_path = dic_path
        self.tfidf_model_path = tfidf_model_path
        self.tfidf_index_path = tfidf_index_path
        for per_path in [self.dic_path, self.tfidf_model_path, self.tfidf_index_path]:
            per_path = '/'.join(per_path.split('/')[:-1])
            if os.path.exists(per_path) == False:
                os.makedirs(per_path)
        self.stop_word = stop_word() 
        self.vectorizer = CountVectorizer(stop_words = None, max_df=cfg.emb.MAX_DF, min_df=cfg.emb.MIN_DF, max_features=cfg.emb.MAX_FEATURES, token_pattern='(?u)\\b\\w\\w*\\b')    
        self.transformer = TfidfTransformer()

    # init
    def init(self, words_list=None, update=True):
        if (~os.path.exists(self.dic_path) or ~os.path.exists(self.tfidf_model_path) or update) and (words_list!=None):
            word_list = self._seg_word(words_list)
            # print ('>>>>>>>>>>', word_list)
        
        if os.path.exists(self.dic_path) and os.path.exists(self.tfidf_model_path) and update==False:
            with open(self.dic_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(self.tfidf_model_path, 'rb') as f:
                self.transformer = pickle.load(f)
        else:
            try:
                 logging.info('[Tfidf] start build tfidf model.')
                 if words_list==None:
                     logging.error( '[Bow] words_list is None' )
                 self._gen_model(word_list)
                 logging.info('[Tfidf] build tfidf model success.')
            except Exception as e:
                 logging.error( '[Tfidf] build tfidf model error，error info: {} '.format(e) )
        
        if words_list!=None:
            self.words_list_pre = []
            for per_word in words_list:
                self.words_list_pre.append( self._normalize( self._predict(per_word) )[0] )
            self.words_list_pre = np.array(self.words_list_pre)
        return self
    
    # gen 3-gram
    def _list_3_ngram(self, words, n=3, m=2):
        pattern1 = re.compile(r'[0-9]')
        if len(words) < n:
            n = len(words)
        temp=[words[i - k:i] for k in range(m, n + 1) for i in range(k, len(words) + 1) ]
        return [item for item in temp if len(''.join(item).strip())>0 and len(pattern1.findall(''.join(item).strip()))==0]
    
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
        # return [ ' '.join(word) for word in word_list  ]
        return [ ' '.join(['_'.join(i) for i in self._list_3_ngram(word,n=3, m=2)]) for word in word_list]


    def fit(self, word_list):
        word_list = self._seg_word(word_list)
        self._gen_model(word_list)
    
    # build dic
    def _gen_dic(self, word_list):
        dic = self.vectorizer.fit_transform(word_list)
        with open(self.dic_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        return dic

    # build tf-idf model
    def _gen_model(self, word_list):
        tfidf = self.transformer.fit_transform(self._gen_dic(word_list))
        with open(self.tfidf_model_path, 'wb') as f:
            pickle.dump(self.transformer, f)
    
    def _predict(self, words):
        tf_idf_embedding = self.transformer.transform( self.vectorizer.transform(self._seg_word([words])) )
        # print('tf_idf_embedding0>>>>>', tf_idf_embedding)
        tf_idf_embedding = tf_idf_embedding.toarray().sum(axis=0)
        # print ('>>>>', tf_idf_embedding[np.newaxis, :]) 
        return tf_idf_embedding[np.newaxis, :].astype(float)
    
    def predict(self, words):
        pre = self._normalize( self._predict(words) )
        return np.dot( self.words_list_pre[:], pre[0] ) 


