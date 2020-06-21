# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  bm25测试
   Author :       machinelp
   Date :         2020-06-11
-------------------------------------------------

'''

import sys
from textmatch.models.text_search.bm25 import BM25
from textmatch.config.constant import Constant as const



if __name__ == '__main__':
    # 存放问句
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]

    bm25 = BM25()
    bm25.init(words_list, update=True)
    testword = "我在九寨沟,很喜欢"
    pre = bm25.predict(testword)
    print ('pre>>>>>', pre) 





