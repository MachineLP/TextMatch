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
from textmatch.models.text_embedding.model_factory_sklearn import ModelFactory
from textmatch.models.text_classifier.dnn import DNN 


class IntentClassifier():

    def __init__( self, model_factory=ModelFactory, match_models=['bow'] ):
        self.model_factory = model_factory
        self._init_model( self.words_dict,  match_models)

    def _init_model(self, match_models=['bow', 'tfidf', 'ngram_tfidf']):
        self.mf = self.model_factory( match_models=words_dict )
    
    def predict(self, words, word_id=None):
        return self.model.predict_emb(words, word_id)






