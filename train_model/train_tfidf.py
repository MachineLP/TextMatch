# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  tf-idf train
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------

'''

import sys
from textmatch.models.text_embedding.tf_idf_sklearn import TfIdf
from textmatch.config.constant import Constant as const


if __name__ == '__main__':
    # 训练集
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    # doc
    words_list1 = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟", "哈哈哈哈"]

    # 训练
    tfidf = TfIdf(dic_path=const.TFIDF_DIC_PATH, tfidf_model_path=const.TFIDF_MODEL_PATH, tfidf_index_path=const.TFIDF_INDEX_PATH, )
    tfidf.fit(words_list)

    # query
    tfidf = TfIdf(dic_path=const.TFIDF_DIC_PATH, tfidf_model_path=const.TFIDF_MODEL_PATH, tfidf_index_path=const.TFIDF_INDEX_PATH, )
    tfidf.init(words_list1, update=False)

    testword = "我在九寨沟,很喜欢"
    pre = tfidf.predict(testword)
    print ('pre>>>>>', pre) 
    # pre>>>>> [0.21092879 0.4535442  0.87695613 0.        ]

    pre = tfidf._predict(testword)[0]
    print ('pre>>>>>', pre) 
    # pre>>>>> [0.63174505 0.         0.4804584  0.4804584  0.         0.37311881          0.        ]


