# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  TextEmbedding 测试
   Author :       machinelp
   Date :         2020-06-06
-------------------------------------------------

'''

import sys
import json 
from textmatch.config.constant import Constant as const
from textmatch.core.text_embedding import TextEmbedding

test_dict = {"id0": "其实事物发展有自己的潮流和规律",
   "id1": "当你身处潮流之中的时候，要紧紧抓住潮流的机会",
   "id2": "想办法脱颖而出，即使没有成功，也会更加洞悉时代的脉搏",
   "id3": "收获珍贵的知识和经验。而如果潮流已经退去",
   "id4": "这个时候再去往这个方向上努力，只会收获迷茫与压抑",
   "id5": "对时代、对自己都没有什么帮助",
   "id6": "但是时代的浪潮犹如海滩上的浪花，总是一浪接着一浪，只要你站在海边，身处这个行业之中，下一个浪潮很快又会到来。你需要敏感而又深刻地去观察，略去那些浮躁的泡沫，抓住真正潮流的机会，奋力一搏，不管成败，都不会遗憾。"}


if __name__ == '__main__':
    # ['bow', 'tfidf', 'ngram_tfidf', 'bert']
    # ['bow', 'tfidf', 'ngram_tfidf', 'bert', 'w2v']
    # text_embedding = TextEmbedding( match_models=['bow', 'tfidf', 'ngram_tfidf', 'w2v'], words_dict=None, update=False ) 
    text_embedding = TextEmbedding( match_models=['bow', 'tfidf', 'ngram_tfidf', 'w2v'], words_dict=test_dict ) 
    pre = text_embedding.predict( "其实事物发展有自己的潮流和规律" ) 
    print ('text_embedding>>>>>', pre) 

    pre = text_embedding.predict( "其实事物发展有自己的潮流和规律", "id1" ) 
    print ('text_embedding>>>>>', pre) 




