# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  text search (召回和排序)
   Author :       machinelp
   Date :         2020-07-19
-------------------------------------------------

'''

import sys
from textmatch.core.text_match import TextMatch
from textmatch.core.qa_match import QMatch, AMatch, SemanticMatch
from textmatch.models.text_search.model_factory_search import ModelFactorySearch



def text_match_recall(testword, doc_dict):
    # QMatch
    q_match = QMatch( q_dict=doc_dict, match_models=['bow', 'tfidf', 'ngram_tfidf', 'albert']) 
    q_match_pre = q_match.predict(testword, match_strategy='score', vote_threshold=0.5, key_weight = {'bow': 1, 'tfidf': 1, 'ngram_tfidf': 1, 'albert': 1})
    # print ('q_match_pre>>>>>', q_match_pre )
    return q_match_pre 

def text_match_sort(testword, candidate_doc_dict):
    text_match = TextMatch( q_dict=candidate_doc_dict, match_models=['bm25', 'edit_sim', 'jaccard_sim'] )
    text_match_res = text_match.predict( query, match_strategy='score', vote_threshold=-100.0, key_weight = {'bm25': 0, 'edit_sim': 1, 'jaccard_sim': 1} )
    return text_match_res 


if __name__ == '__main__':
    doc_dict = {"0":"我去玉龙雪山并且喜欢玉龙雪山玉龙雪山", "1":"我在玉龙雪山并且喜欢玉龙雪山", "2":"我在九寨沟", "3":"我在九寨沟,很喜欢", "4":"很喜欢"}   
    query = "我在九寨沟,很喜欢"
    
    # 
    mf = ModelFactorySearch( match_models=['bm25', 'edit_sim', 'jaccard_sim'] )
    mf.init(words_dict=doc_dict)
    pre = mf.predict(query)
    print ('pre>>>>>', pre) 

    
    # 召回
    match_pre = text_match_recall( query, doc_dict )
    print( '召回的结果:', match_pre )

    candidate_doc_dict = dict( zip( match_pre.keys(), [doc_dict[key] for key in match_pre.keys()] ) )
    print ("candidate_doc_dict:", candidate_doc_dict)

    '''
    # 排序
    mf = ModelFactorySearch( match_models=['bm25', 'edit_sim', 'jaccard_sim'] )
    mf.init(words_dict=candidate_doc_dict) 
    pre = mf.predict(query)
    print ('排序的结果>>>>>', pre) 
    '''


    # 排序
    # ['bm25', 'edit_sim', 'jaccard_sim']
    text_match_res = text_match_sort( query, candidate_doc_dict )
    print ('排序的score>>>>>', text_match_res) 


