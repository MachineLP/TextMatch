# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  albert_embedding测试
   Author :       machinelp
   Date :         2020-06-18
-------------------------------------------------

'''

import sys
from textmatch.models.text_embedding.albert_embedding import ALBertEmbedding
from textmatch.config.constant import Constant as const


if __name__ == '__main__':
    words_list = ["我是他老公有什么事，你跟我讲就行了。", "我需要提前结清", "你们是哪里，你们是谁？"]
    albert = ALBertEmbedding(config_path=const.ALBERT_CONFIG_PATH, albert_checkpoint_path = const.ALBERT_CHECKPOINT_PATH, dict_path = const.ALBERT_DICT_PATH )
    albert.init(words_list)
    testword = "我是他家人/朋友，你有什么事可以给我说？"
    pre = albert.predict(testword)
    print ('pre>>>>>', pre) 
    # pre>>>>> [0.71687514 0.39825273 0.4128961 ]
