# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  bert train
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''
import numpy as np
import pandas as pd
from keras.layers import *
from bert4keras.backend import keras, set_gelu
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizer import Tokenizer
from textmatch.config.constant import Constant as const
from textmatch.models.text_embedding.bert_embedding import BertEmbedding

set_gelu('tanh')  # 切换gelu版本

maxlen = 32
batch_size = 16
num_classes = 2
epochs = 20
learning_rate = 2e-5 


# sim roeberta_zh
# 【百度网盘】链接:https://pan.baidu.com/s/1RVAHqL1CfLGltPWpoTyThw  密码:bp71
config_path = 'publish/bert_config.json'
checkpoint_path = 'publish/bert_model.ckpt'
dict_path = 'publish/vocab.txt'


def load_data(filename):
    D = []
    data = pd.read_csv(filename)
    data.dropna(axis=0, how='any', inplace=True)
    data = data.values.tolist()
    for per_data in data:
        D.append( (per_data[0],per_data[1],int(per_data[2])) )
    return D


# 加载数据集
train_val_data = load_data('./data/train_data.csv') 
# test_data = load_data('dev.csv') 
# 查看一下数据
print ( 'train>>>>', train_val_data[0] )
print ( '训练验证集数量:', len(train_val_data) )

random_order = range(len(train_val_data))
np.random.shuffle(list(random_order))
train_data = [train_val_data[j] for i, j in enumerate(random_order) if i % 5 != 1 ] 
valid_data = [train_val_data[j] for i, j in enumerate(random_order) if i % 5 == 1 ] 
test_data = valid_data
print ( '训练集数量:', len(train_data) )
print ( '验证集数量:', len(valid_data) )
print ( '测试集数量:', len(test_data) )



# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text1, text2, label = self.data[i]
#             print(text1, text2, label)
            token_ids, segment_ids = tokenizer.encode(text1, text2, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

const.BERT_CONFIG_PATH = config_path
const.BERT_CHECKPOINT_PATH = checkpoint_path
const.BERT_DICT_PATH = dict_path

bert_embedding = BertEmbedding(const.BERT_CONFIG_PATH, const.BERT_CHECKPOINT_PATH, const.BERT_DICT_PATH, train_mode=True)
bert = bert_embedding.bert
output = Dropout(rate=0.1)(bert.model.output)
output = Dense(units=num_classes,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_bert_model.weights')
        test_acc = evaluate(test_generator)
        print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n'
              % (val_acc, self.best_val_acc, test_acc))

evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    callbacks=[evaluator])

model.load_weights('best_bert_model.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))


