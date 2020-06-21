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
from textmatch.models.text_embedding.albert_embedding import ALBertEmbedding

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

const.ALBERT_CONFIG_PATH = config_path
const.ALBERT_CHECKPOINT_PATH = checkpoint_path
const.ALBERT_DICT_PATH = dict_path
# 加载预训练模型
bert_embedding = ALBertEmbedding(const.ALBERT_CONFIG_PATH, const.ALBERT_CHECKPOINT_PATH, const.ALBERT_DICT_PATH, train_mode=True)
bert = bert_embedding.bert

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(units=num_classes,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
model.load_weights('best_model.weights', by_name=True)


def sim_words(word1, word2):
    token_ids1, segment_ids1 = tokenizer.encode( word1, word2 )
    pre1 = model.predict([np.array([token_ids1, token_ids1]), np.array([segment_ids1, segment_ids1])])
    return pre1


word1 = '我能延迟几天还花呗吗' 
word2 = '为什么暂时不能开通商家花呗收款' 
print( ">>>>", sim_words(word1, word2) )


