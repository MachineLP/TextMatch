# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  CNN   
   Author :       machinelp
   Date :         2020-06-26
-------------------------------------------------

'''

import os
import gc
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, Model, Input
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, \
    concatenate, \
    Activation, ZeroPadding2D, Lambda, Embedding, Permute, Concatenate
from keras.layers import add, Flatten
from keras.callbacks import Callback


def Conv2d_BN(x, nb_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x

#网络构造
def Net_2d(l=10, cols_len=10):
    #emb_size = 6
    input_1 = Input(shape=(l, cols_len, 1), name='input_1')
    input_1_1 = Lambda(lambda x: x[:, :, 0:, :])(input_1)  #获取除了particle_category其他特征 
    #input_1_2 = Lambda(lambda x: x[:, :, 0, :])(input_1)  #获取particle_category
    #cate_emb = Embedding(output_dim=emb_size, input_dim=15)(input_1_2)  #将particle_category转换成Embedding
    #cate_emb = Permute((1, 3, 2))(cate_emb)
    #X = Concatenate(axis=2)([input_1_1, cate_emb])  #拼接particle_category的Embedding 跟 其他特征
    X = input_1_1
    X = Conv2d_BN(X, nb_filter=64)
    shortcut = Conv2d_BN(X, nb_filter=128, kernel_size=(1, cols_len - 1))
    X = Conv2d_BN(X, nb_filter=128, kernel_size=(1, cols_len - 1))
    X = Conv2d_BN(X, nb_filter=256)
    X = Conv2d_BN(X, nb_filter=512)
    X = Concatenate(axis=-1)([X, shortcut])
    X = Conv2d_BN(X, nb_filter=128)
    X = Conv2d_BN(X, nb_filter=256)
    X = Conv2d_BN(X, nb_filter=512)
    # X = Conv2d_BN(X, nb_filter=128, kernel_size=(8, 1), padding='same')
    X = AveragePooling2D(pool_size=(l, 1))(X)
    X = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(X))))
    X = Dense(19, activation='softmax')(X)
    model = Model([input_1], X)

    return model




# 网络构造
def Net_1d(l=50, cols_len=8, filter_sizes=[1, 2, 3]):
    input_1 = Input(shape=(l, cols_len), name='input_1')
    cnn1 = Conv1D(256, filter_sizes[0], padding='same', strides=1, activation='relu')(input_1)
    cnn1 = BatchNormalization(axis=2, name='bn1')(cnn1)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, filter_sizes[1], padding='same', strides=1, activation='relu')(input_1)
    cnn2 = BatchNormalization(axis=2, name='bn2')(cnn2)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256, filter_sizes[2], padding='same', strides=1, activation='relu')(input_1)
    cnn3 = BatchNormalization(axis=2, name='bn3')(cnn3)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    X = Dense(19, activation='softmax')(drop)
    model = Model([input_1], X)

    return model


