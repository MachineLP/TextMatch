# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  tf-idf实现
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------

'''

import sys
from textmatch.models.text_embedding.ngram_tf_idf_sklearn import NgramTfIdf
from textmatch.config.constant import Constant as const


if __name__ == '__main__':
    # 训练集
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    # doc
    words_list1 = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟", "哈哈哈哈"]

    # 训练
    tfidf = NgramTfIdf(dic_path=const.NGRAM_TFIDF_DIC_PATH, tfidf_model_path=const.NGRAM_TFIDF_MODEL_PATH, tfidf_index_path=const.NGRAM_TFIDF_INDEX_PATH, )
    tfidf.fit(words_list)

    # query
    tfidf = NgramTfIdf(dic_path=const.NGRAM_TFIDF_DIC_PATH, tfidf_model_path=const.NGRAM_TFIDF_MODEL_PATH, tfidf_index_path=const.NGRAM_TFIDF_INDEX_PATH, )
    tfidf.init(words_list1, update=False)
    testword = "我在九寨沟,很喜欢"
    #for word in jieba.cut(testword):
    #    print ('>>>>', word)
    pre = tfidf.predict(testword)
    print ('pre>>>>>', pre) 

    pre = tfidf._predict(testword)[0]
    print ('pre>>>>>', pre) 


