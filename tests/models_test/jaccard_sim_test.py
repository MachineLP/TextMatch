# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  Jaccard sim 测试
   Author :       machinelp
   Date :         2020-06-11
-------------------------------------------------

'''

import sys
from textmatch.models.text_search.jaccard_sim import Jaccard
from textmatch.config.constant import Constant as const



if __name__ == '__main__':
    # 存放问句
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]

    jaccard_dis = Jaccard()
    jaccard_dis.init(words_list)
    testword = "我在九寨沟,很喜欢"
    pre = jaccard_dis.predict(testword)
    print ('pre>>>>>', pre) 





