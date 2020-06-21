# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  bert_embedding实现
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import numpy as np
import tensorflow as tf
from textmatch.config.constant import Constant as const
from textmatch.models.model_base.model_base import ModelBase
from bert4keras.backend import keras, set_gelu
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer
from bert4keras.snippets import sequence_padding
set_gelu('tanh')  

class ALBertEmbedding(ModelBase):
    '''通过ALBert计算句向量
    '''
    def __init__(self, 
                       config_path=const.ALBERT_CONFIG_PATH, 
                       albert_checkpoint_path = const.ALBERT_CHECKPOINT_PATH, 
                       dict_path = const.ALBERT_DICT_PATH,
                       train_mode=False ):
        self.session = tf.Session() 
        keras.backend.set_session(self.session)
        if train_mode:
            self.bert = build_bert_model(
                         model='albert', 
                         config_path=config_path,
                         checkpoint_path=albert_checkpoint_path,
                         with_pool=True,
                         return_keras_model=False,)
        else:
            self.bert = build_bert_model(
                         model='albert', 
                         config_path=config_path,
                         # checkpoint_path=albert_checkpoint_path,
                         with_pool=True,
                         return_keras_model=False,)
            self.encoder = keras.models.Model(self.bert.model.inputs, self.bert.model.outputs[0])
            self.tokenizer = Tokenizer(dict_path, do_lower_case=True) 
            self.encoder.load_weights(albert_checkpoint_path, by_name=True)
    
    def init(self, words_list=None, update=True):
        if words_list!=None:
            token_ids_list, segment_ids_list = [], []
            for words in words_list:
                token_ids, segment_ids = self.tokenizer.encode(words)
                token_ids_list.append(token_ids)
                segment_ids_list.append(segment_ids)
            token_ids_list = sequence_padding(token_ids_list)
            segment_ids_list = sequence_padding(segment_ids_list)
            self.words_list_pre = self.encoder.predict([token_ids_list, segment_ids_list])
            self.words_list_pre = self._normalize(self.words_list_pre)
        return self
    
    def _predict(self, words):
        with self.session.as_default():
            with self.session.graph.as_default():
                token_ids, segment_ids = self.tokenizer.encode( words )
                pre = self.encoder.predict([np.array([token_ids]), np.array([segment_ids])])
                pre = self._normalize(pre)
        return pre
        
    # 句向量 
    def predict(self, words):
        with self.session.as_default():
            with self.session.graph.as_default():
                token_ids, segment_ids = self.tokenizer.encode( words )
                pre = self.encoder.predict([np.array([token_ids]), np.array([segment_ids])])
                pre = self._normalize(pre)
        return np.dot( self.words_list_pre[:], pre[0] ) 


