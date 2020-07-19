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
from textmatch.models.text_search.model_factory_search import ModelFactorySearch


class TextMatchBase():
    def __init__(self, model_factory=ModelFactorySearch):
        self.model_factory = model_factory

    def _init_model(self, q_dict, match_models):
        self.mf = self.model_factory( match_models=match_models )
        self.mf.init(words_dict=q_dict, update=True)

    def _predict(self, words):
        return self.mf.predict(words)
    
    def predict():
        pass

    def _normalize(self, x, key_weight):
        return x / float(np.sum( list(key_weight.values()) ))

    def vote(self, res, vote_threshold=0.75, key_weight = {'bm25': 1, 'edit_sim': 1, 'jaccard_sim': 1}):
        qa_hit = {}
        for key, value in res.items():   # { 'bow':['q1':0.5, 'q2':0.3...], 'tf-idf'....} 
            for qa_id, qa_score in value: 
                if qa_id not in qa_hit.keys(): 
                    qa_hit[qa_id] = 0 
                if qa_score>0.5: 
                    qa_hit[qa_id] += 1 * key_weight[key] 
        qa_res = {}
        for qa_id, qa_score in qa_hit.items():
            qa_score = self._normalize( qa_score,key_weight )
            qa_hit[qa_id] = qa_score
            if qa_score>=vote_threshold:
                qa_res[qa_id] = qa_score
        return [qa_res, qa_hit]

    def score(self, res, score_threshold=0.625, key_weight = {'bm25': 1, 'edit_sim': 1, 'jaccard_sim': 1}):
        qa_hit = {}
        for key,value in res.items():   # { 'bow':['q1':0.5, 'q2':0.3...], 'tf-idf'....} 
            for qa_id, qa_score in value: 
                if qa_id not in qa_hit.keys(): 
                    qa_hit[qa_id] = 0 
                qa_hit[qa_id] +=  qa_score * key_weight[key] 
        qa_res = {}
        for qa_id,qa_score in qa_hit.items():
            qa_score = self._normalize( qa_score,key_weight )
            qa_hit[qa_id] = qa_score
            if qa_score>=score_threshold:
                qa_res[qa_id] = qa_score
        return [qa_res, qa_hit]

class TextMatch(TextMatchBase):
    '''用于问句句匹配
        input: words
        output: {'id0':0.2, 'id1':0.5, ...}
    '''
    def __init__(self,  q_dict, model_factory=ModelFactorySearch, match_models=['bm25', 'edit_sim', 'jaccard_sim']):
        super().__init__(model_factory)
        self.q_dict = q_dict
        self._init_model( self.q_dict,  match_models=match_models ) 

    def predict(self, words, match_strategy='score', vote_threshold=0.75, key_weight = {'bm25': 1, 'edit_sim': 1, 'jaccard_sim': 1}):
        res = self._predict( words )
        if match_strategy == 'vote':
            return self.vote(res, vote_threshold, key_weight)[0]
        if match_strategy == 'score':
            return self.score(res, vote_threshold, key_weight)[0]
