# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QA测试
   Author :       machinelp
   Date :         2019-03-31
-------------------------------------------------

'''

import sys
from textmatch.core.qa_match import QMatch, AMatch, SemanticMatch

test_dict = {"id0": "其实事物发展有自己的潮流和规律",
   "id1": "当你身处潮流之中的时候，要紧紧抓住潮流的机会",
   "id2": "想办法脱颖而出，即使没有成功，也会更加洞悉时代的脉搏",
   "id3": "收获珍贵的知识和经验。而如果潮流已经退去",
   "id4": "这个时候再去往这个方向上努力，只会收获迷茫与压抑",
   "id5": "对时代、对自己都没有什么帮助",
   "id6": "但是时代的浪潮犹如海滩上的浪花，总是一浪接着一浪，只要你站在海边，身处这个行业之中，下一个浪潮很快又会到来。你需要敏感而又深刻地去观察，略去那些浮躁的泡沫，抓住真正潮流的机会，奋力一搏，不管成败，都不会遗憾。"}


def test_q_match(testword):
    # QMatch
    q_match = QMatch( q_dict=test_dict, match_models=['bow', 'tfidf', 'ngram_tfidf']) 
    q_match_pre = q_match.predict(testword, match_strategy='score', vote_threshold=0.5, key_weight = {'bow': 1, 'tfidf': 1, 'ngram_tfidf': 1})
    print ('q_match_pre>>>>>', q_match_pre )
    return q_match_pre

def test_a_match(testword):
    # AMatch
    a_match = AMatch( a_dict=test_dict, match_models=['bow', 'tfidf', 'ngram_tfidf']) 
    a_match_pre = a_match.predict(testword, ['id0', 'id1'], match_strategy='score', vote_threshold=0.5, key_weight = {'bow': 1, 'tfidf': 1, 'ngram_tfidf': 1}) 
    print ('a_match_pre>>>>>', a_match_pre )
    # a_match_pre>>>>> {'id0': 1.0, 'id1': 0.0} 
    return a_match_pre


def test_semantic_match(testword,words_dict=test_dict):
    # SemanticMatch
    s_match = SemanticMatch( words_dict=words_dict, match_models=['bow', 'tfidf', 'ngram_tfidf'] ) 
    s_match_pre = s_match.predict(testword, ['id0','id1', "id5"], match_strategy='score', vote_threshold=0.5, key_weight = {'bow': 1, 'tfidf': 1, 'ngram_tfidf': 1})
    print ('s_match_pre>>>>>', s_match_pre ) 
    # s_match_pre>>>>> {'id0': 1.0, 'id1': 0.0}
    return s_match_pre




if __name__ == '__main__':
    testword = "其实事物发展有自己的潮流和规律"
    test_q_match(testword)
    test_a_match(testword)
    test_semantic_match(testword)
    





