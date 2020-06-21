# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  w2v 训练
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------

'''
import os
import time
import jieba
import gensim
import threading
import numpy as np
from textmatch.config.config import Config as conf
from textmatch.config.constant import Constant as const
from textmatch.models.text_embedding.stop_words import StopWords


# min_count,频数阈值，大于等于1的保留
# size，神经网络 NN 层单元数，它也对应了训练算法的自由程度
# workers=4，default = 1 worker = no parallelization 只有在机器已安装 Cython 情况下才会起到作用。如没有 Cython，则只能单核运行。

if __name__ == '__main__':
    stop_word = StopWords(stopwords_file=const.STOPWORDS_FILE)
    # 训练集
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟", "其实事物发展有自己的潮流和规律，当你身处潮流之中的时候，要紧紧抓住潮流的机会，想办法脱颖而出，即使没有成功，也会更加洞悉时代的脉搏，收获珍贵的知识和经验。而如果潮流已经退去，这个时候再去往这个方向上努力，只会收获迷茫与压抑，对时代、对自己都没有什么帮助。", "但是时代的浪潮犹如海滩上的浪花，总是一浪接着一浪，只要你站在海边，身处这个行业之中，下一个浪潮很快又会到来。你需要敏感而又深刻地去观察，略去那些浮躁的泡沫，抓住真正潮流的机会，奋力一搏，不管成败，都不会遗憾。"]
    # doc
    # words_list1 = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟", "哈哈哈哈"]
    del_stopword = True
    corpus = []
    for per_words in words_list:
        word_list = jieba.cut(per_words,cut_all=False)
        if del_stopword:
            word_list = stop_word.del_stopwords(word_list)
        corpus.append(word_list)
    
    # min_count,频数阈值，大于等于1的保留
    # size，神经网络 NN 层单元数，它也对应了训练算法的自由程度
    # workers=4，default = 1 worker = no parallelization 只有在机器已安装 Cython 情况下才会起到作用。如没有 Cython，则只能单核运行。
    model = gensim.models.Word2Vec(corpus, min_count=1, size=256)
    for per_path in [const.W2V_MODEL_FILE]:
        per_path = '/'.join(per_path.split('/')[:-1])
        if os.path.exists(per_path) == False:
            os.makedirs(per_path)
    model.save(const.W2V_MODEL_FILE)

    vector = model[corpus[-2]]
    print('vector>>>>', vector)








