# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QA kw测试
   Author :       machinelp
   Date :         2020-07-18
-------------------------------------------------

'''

import sys
from textmatch.core.qa_match_kw import QAMatchKW

res_dict={'id0':0.8, 'id1':0.3}

qkw_dict = {
    'id0': ['神仙水|神仙', '价格|多少钱'],
    'id1': ['海蓝之谜|lammer', '面霜', '功效|功能|作用'],
    'id2': ['快递']
    }

akw_dict = {
    'id0': ['799|七百九十九|七九九'],
    'id1': ['补水|祛斑'],
    'id2': ['顺丰']
    }

def test_qkw_match(testword):
    qkw_match = QAMatchKW( qkw_dict=qkw_dict, akw_path=akw_dict ) 
    res = qkw_match.post_processing_q( testword,res_dict )
    print ('res>>>>>', res )
    return res


def test_akw_match(testword):
    qkw_match = QAMatchKW( qkw_dict=qkw_dict, akw_path=akw_dict ) 
    res = qkw_match.post_processing_a( testword,res_dict )
    print ('res>>>>>', res )
    return res



if __name__ == '__main__':
    testword = "神仙税多少钱"
    test_qkw_match(testword)
    testword = "799"
    test_akw_match(testword)

'''
res>>>>> {'id0': 1.0}
res>>>>> {'id0': [1.0, '神仙水'], 'id1': [0.0, '海蓝之谜|lammer']}
'''
