# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  text_encoding测试
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

from textmatch.models.text_encoding.text_hash import TextEncoding



if __name__ == '__main__':
    text_enc = TextEncoding()
    test_res = text_enc.get_hash("你好啊")
    print (">>>>>", test_res)

    test_res = text_enc.get_sim_simhash('在这里干嘛呢有时间吗一起吃个饭吧', '你在这里干嘛呢有时间吗一起吃个饭吧')
    print (">>>>>", test_res)





