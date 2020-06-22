# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QAMatchBase 
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import json 
import numpy as np
from textmatch.models.text_embedding.model_factory_sklearn import ModelFactory as ModelFactory_emb
from textmatch.models.text_classifier.model_factory import ModelFactory as ModelFactory_cls


class IntentClassifier():

    def __init__( self, model_factory_emb=ModelFactory_emb, model_factory_cls=ModelFactory_cls, match_models=['bow'] ):
        self.model_factory_emb = model_factory_emb
        self.model_factory_cls = model_factory_cls
        self._init_model( self.words_dict,  match_models)

    def _init_model(self, match_models=['bow', 'tfidf', 'ngram_tfidf']):
        self.mf_emb = self.model_factory_emb( match_models=words_dict ) 
        self.mf_cls = self.model_factory_cls( match_models=words_dict_ ) 
    
    def predict_emb(self, words, word_id=None):
        return self.mf_emb.predict_emb(words, word_id)
    
    def predict_cls(self, words, word_id=None):
        return self.mf_cls.predict_cls(words)


    def predict(self, words, word_id=None):
        emb = self.predict_emb( words ) 
        words_emb=[]
        for per_v in emb.values():
           words_emb.extend( per_v )
        words_emb = np.array(words_emb)
        return self.predict_cls( words_emb ) 






