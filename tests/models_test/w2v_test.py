# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  w2v测试
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import sys
from textmatch.models.text_embedding.w2v import Word2Vec
from textmatch.models.text_embedding.stop_words import StopWords
from textmatch.config.constant import Constant as const



if __name__ == '__main__':
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    w2v = Word2Vec(w2v_model_file=const.W2V_MODEL_FILE,  stop_word=StopWords(stopwords_file=const.STOPWORDS_FILE) )
    w2v.init(words_list, update=True)
    testword = "我在九寨沟,很喜欢"
    pre = w2v.predict(testword)
    print ('pre>>>>>', pre) 


