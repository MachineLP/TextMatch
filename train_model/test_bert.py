# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  albert train
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------

'''

from keras.layers import *

from bert4keras.backend import keras, set_gelu
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizer import Tokenizer
import pandas as pd
import numpy as np
from textmatch.config.constant import Constant as const
from textmatch.models.text_embedding.bert_embedding import BertEmbedding

set_gelu('tanh')  # 切换gelu版本

maxlen = 32
batch_size = 16
num_classes = 2
epochs = 20
learning_rate = 2e-5 

config_path = 'albert_tiny_google_zh_489k/albert_config.json'
checkpoint_path = 'albert_tiny_google_zh_489k/albert_model.ckpt'
dict_path = 'albert_tiny_google_zh_489k/vocab.txt'


tokenizer = Tokenizer(dict_path, do_lower_case=True) # 建立分词器
# 加载预训练模型
const.BERT_CONFIG_PATH = config_path
const.BERT_CHECKPOINT_PATH = checkpoint_path
const.BERT_DICT_PATH = dict_path

bert_embedding = BertEmbedding(const.BERT_CONFIG_PATH, const.BERT_CHECKPOINT_PATH, const.BERT_DICT_PATH, train_mode=True)
bert = bert_embedding.bert

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
encoder.load_weights('best_model.weights', by_name=True)


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = cos
    # sim = 0.5 + 0.5 * cos
    return sim
# encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
# 输入两个句子输出相似度
def sim_words(word1, word2):
    token_ids1, segment_ids1 = tokenizer.encode( word1 )
    token_ids2, segment_ids2 = tokenizer.encode( word2 )
    pre1 = encoder.predict([np.array([token_ids1]), np.array([segment_ids1])])
    pre2 = encoder.predict([np.array([token_ids2]), np.array([segment_ids2])])
    print ('>>>>>', pre1)
    # pre[0], pre[1]   pre[0].mean(axis=0)
    return cos_sim( pre1[0], pre2[0] )


word1 = '我的花呗不能用了' 
word2 = '我的蚂蚁花呗分期不可用' 

print( ">>>>", sim_words(word1, word2) )



