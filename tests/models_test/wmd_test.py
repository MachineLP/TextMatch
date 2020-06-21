# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  ModelFactory 测试
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------

'''

import time
import jieba
import gensim
import threading
import numpy as np
from textmatch.config.constant import Constant as const
# 粗排：使用word mover distance（WMD）来进行初始的排查，最终得分0-0.15的太相似了，0.45-1分的基本不相关，所以从0.15-0.45分钟选择了10%来进行人工标注


# word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
w2v_model_file = const.W2V_MODEL_FILE
w2v_model = gensim.models.Word2Vec.load(w2v_model_file)

w2v_model.init_sims(replace=True) # normalizes vectors
distance = w2v_model.wmdistance("你们是你们哪，你们哪里的。", "你们是哪里，你们是谁？")  
print ('distance>>>>', distance) 



'''
"你有什么事你说。", "我是他家人/朋友，你有什么事可以给我说？"            0.6694891459671026
"呃，我想提前结清我名下那个款项。", "我需要提前结清"                    0.6992085239002946
"你们是你们哪，你们哪里的。", "你们是哪里，你们是谁？"                  0.27438064142232443   
"嗯，好。", "你们催收人员说要对我上门催收，是不是真的？"                 0.948713353219643
"嗯。就是您就是就是。就是您就是您拨打的这个电话。", "你们催收人员说要对我上门催收，是不是真的？"                 0.8855274054486878
"提前结清。", "我需要提前结清"                 0.5150805852253076
'''
