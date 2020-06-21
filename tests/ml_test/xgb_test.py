# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  gbdt lr test
   Author :       machinelp
   Date :         2020-06-17
-------------------------------------------------

'''
import sys
import numpy as np
from textmatch.models.text_search.bm25 import BM25
from textmatch.config.constant import Constant as const
from textmatch.models.ml.xgb import XGB

if __name__ == '__main__':
    # 构造训练数据

    train_x = np.ones([100, 10])
    train_y = np.append( np.ones([50, 1]), np.zeros([50, 1]) )
    print ('>>>', train_y)
    xgb = XGB()
    xgb.fit( train_x, train_y )
    res = xgb.predict(train_x)
    print ('>>>>', res)


