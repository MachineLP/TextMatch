# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  TextEmbedding
   Author :       machinelp
   Date :         2020-06-06
-------------------------------------------------

'''

import sys
import json 
from textmatch.config.constant import Constant as const
from textmatch.models.text_embedding.model_factory_sklearn import ModelFactory

class TextEmbedding():
    '''
    model=ModelFactory( match_models=['bow', 'tfidf', 'ngram_tfidf', 'bert', 'w2v']
    输出：{'bow':vector, 'tfidf':vector, .....}
    or
    输出：{'bow': (vector1, vector2), 'tfidf': (vector1, vector2), .....}
    '''
    def __init__(self, model=ModelFactory, match_models=['bow'], words_dict=None, update=True):
        self.model = model
        self.words_dict = words_dict
        self._init_model( self.words_dict,  match_models, update=update)

    def _init_model(self, words_dict, match_models, update=True):
        self.mf = self.model( match_models=match_models )
        self.mf.init(words_dict=words_dict, update=update)
   
    def predict(self, words, word_id=None):
        return self.mf.predict_emb(words, word_id)






