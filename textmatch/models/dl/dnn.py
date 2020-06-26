# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  DNN  
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


#网络构造
def Net_nn(cols_len=10):
    input_1 = Input(shape=(cols_len), name='input_1')
    X = Dense(128, activation='relu')(input_1)
    X = Dropout(0.5)(X)
    X = Dense(19, activation='softmax')(X)
    model = Model([input_1], X)

    return model


