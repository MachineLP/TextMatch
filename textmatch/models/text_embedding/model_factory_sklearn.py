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
from textmatch.models.text_embedding.bow_sklearn import Bow
from textmatch.models.text_embedding.tf_idf_sklearn import TfIdf
from textmatch.models.text_embedding.ngram_tf_idf_sklearn import NgramTfIdf
from textmatch.models.text_embedding.w2v import Word2Vec
from textmatch.models.text_embedding.stop_words import StopWords
from textmatch.models.text_embedding.bert_embedding import BertEmbedding
from textmatch.models.text_embedding.albert_embedding import ALBertEmbedding

'''
'''

class ModelFactory(object):
    '''match model factory
    '''
    def __init__(self, match_models=['bow', 'tfidf', 'ngram_tfidf'],
                       bow_model = Bow,
                       tf_idf_model=TfIdf,
                       ngram_tf_idf_model=NgramTfIdf,
                       w2v_model=Word2Vec,
                       #bert_embedding_model=BertEmbedding,
                       albert_embedding_model=ALBertEmbedding,
                       ):
       self.model = {}
       for match_model in match_models:
           if match_model == 'bow':
               model = bow_model(                      dic_path=const.BOW_DIC_PATH, 
                                                       bow_index_path=const.BOW_INDEX_PARH, )
               self.model[match_model] = model
           elif match_model == 'tfidf':
               model = tf_idf_model(                   dic_path=const.TFIDF_DIC_PATH, 
                                                       tfidf_model_path=const.TFIDF_MODEL_PATH, 
                                                       tfidf_index_path=const.TFIDF_INDEX_PATH, )
               self.model[match_model] = model
           elif match_model == 'ngram_tfidf':
               model = ngram_tf_idf_model(             dic_path=const.NGRAM_TFIDF_DIC_PATH, 
                                                       tfidf_model_path=const.NGRAM_TFIDF_MODEL_PATH, 
                                                       tfidf_index_path=const.NGRAM_TFIDF_INDEX_PATH, )
               self.model[match_model] = model
           elif match_model == 'w2v':
               model = w2v_model(                       w2v_model_file=const.W2V_MODEL_FILE, 
                                                        stop_word=StopWords(stopwords_file=const.STOPWORDS_FILE) )
               self.model[match_model] = model
           elif match_model == 'bert':
               model = bert_embedding_model(            config_path=const.BERT_CONFIG_PATH, 
                                                        checkpoint_path = const.BERT_CHECKPOINT_PATH, 
                                                        dict_path = const.BERT_DICT_PATH)
               self.model[match_model] = model
           elif match_model == 'albert':
               model = albert_embedding_model(
                                                        config_path=const.ALBERT_CONFIG_PATH, 
                                                        albert_checkpoint_path = const.ALBERT_CHECKPOINT_PATH, 
                                                        dict_path = const.ALBERT_DICT_PATH, )
                                                        #albert_checkpoint_path = const.ALCHECKPOINT_PATH)
               self.model[match_model] = model
           else:
               logging.error( "[ModelFactory] match_model not existed，please select from ['bow', 'tfidf', 'ngram_tfidf', 'w2v', 'bert', 'albert'] " )
               continue
    
    def init(self, words_dict=None, update=False):
        if words_dict != None:
            self.id_lists, self.words_list = self._dic2list(words_dict)
        else:
            self.id_lists, self.words_list = None, None
        for key, model in self.model.items():
           self.model[key] = model.init(self.words_list, update)

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

    def predict_emb(self, words, word_id=None):
        pre_dict = {}
        for key, model in self.model.items():
            if word_id is not None:
                pre_dic = {}
                pre_list = model.words_list_pre
                for words_id, emb in zip(self.id_lists, pre_list):
                    pre_dic[words_id] = emb
                pre = model._predict(words)[0]
                pre_dict[key] = (pre, pre_dic[word_id])
            else:
                pre = model._predict(words)[0]
                pre_dict[key] = pre
        return pre_dict 
     
    



