# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  bow train
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import sys
from textmatch.models.text_embedding.bow_sklearn import Bow
from textmatch.config.constant import Constant as const



if __name__ == '__main__':
    # 训练集
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    # doc
    words_list1 = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟", "哈哈哈哈"]

    # 训练
    bow = Bow(dic_path=const.BOW_DIC_PATH, bow_index_path=const.BOW_INDEX_PARH, )
    bow.fit(words_list)

    # query
    bow = Bow(dic_path=const.BOW_DIC_PATH, bow_index_path=const.BOW_INDEX_PARH, )
    bow.init(words_list1, update=False)
    testword = "我在九寨沟,很喜欢"
    #for word in jieba.cut(testword):
    #    print ('>>>>', word)
    pre = bow.predict(testword)
    print ('pre>>>>>', pre) 

    pre = bow._predict(testword)[0]
    print ('pre>>>>>', pre) 