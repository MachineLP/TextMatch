
0、添加项目路径
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch

1、训练bow ( python train_model/train_bow.py )
```
from textmatch.models.text_embedding.bow_sklearn import Bow
from textmatch.config.constant import Constant as const

if __name__ == '__main__':
    # 训练集  
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    # doc
    words_list1 = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟", "哈哈哈哈"]
    # 训练
    bow = Bow(dic_path=const.BOW_DIC_PATH, bow_index_path=const.BOW_INDEX_PARH, )
    bow.fit(words_list)

    # query
    bow = Bow(dic_path=const.BOW_DIC_PATH, bow_index_path=const.BOW_INDEX_PARH, )  # 模型保存路径：const.BOW_DIC_PATH、 const.BOW_INDEX_PARH
    bow.init(words_list1, update=False)
    testword = "我在九寨沟,很喜欢"

    pre = bow.predict(testword)
    print ('pre>>>>>', pre) 
    # 输出和doc的相似度
    # [0.27735009 0.53033008 0.86602538 0.        ]

    pre = bow._predict(testword)[0]
    print ('pre>>>>>', pre) 
    # 输出embedding
    # [1. 0. 0. 1. 1. 0. 1. 0.]
```


2、训练tfidf ( python train_model/train_tfidf.py )
```
from textmatch.models.text_embedding.tf_idf_sklearn import TfIdf
from textmatch.config.constant import Constant as const


if __name__ == '__main__':
    # 训练集
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    # doc
    words_list1 = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟", "哈哈哈哈"]

    # 训练
    tfidf = TfIdf(dic_path=const.TFIDF_DIC_PATH, tfidf_model_path=const.TFIDF_MODEL_PATH, tfidf_index_path=const.TFIDF_INDEX_PATH, )
    tfidf.fit(words_list)

    # query
    tfidf = TfIdf(dic_path=const.TFIDF_DIC_PATH, tfidf_model_path=const.TFIDF_MODEL_PATH, tfidf_index_path=const.TFIDF_INDEX_PATH, )
    tfidf.init(words_list1, update=False)

    testword = "我在九寨沟,很喜欢"
    pre = tfidf.predict(testword)
    print ('pre>>>>>', pre) 
    # pre>>>>> [0.21092879 0.4535442  0.87695613 0.        ]

    pre = tfidf._predict(testword)[0]
    print ('pre>>>>>', pre) 
    # pre>>>>> [0.63174505 0.         0.4804584  0.4804584  0.         0.37311881          0.        ]

```


3、训练ngram tfidf ( python train_model/train_ngram_tfidf.py )
```
import sys
from textmatch.models.text_embedding.ngram_tf_idf_sklearn import NgramTfIdf
from textmatch.config.constant import Constant as const


if __name__ == '__main__':
    # 训练集
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    # doc
    words_list1 = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟", "哈哈哈哈"]

    # 训练
    tfidf = NgramTfIdf(dic_path=const.NGRAM_TFIDF_DIC_PATH, tfidf_model_path=const.NGRAM_TFIDF_MODEL_PATH, tfidf_index_path=const.NGRAM_TFIDF_INDEX_PATH, )
    tfidf.fit(words_list)

    # query
    tfidf = NgramTfIdf(dic_path=const.NGRAM_TFIDF_DIC_PATH, tfidf_model_path=const.NGRAM_TFIDF_MODEL_PATH, tfidf_index_path=const.NGRAM_TFIDF_INDEX_PATH, )
    tfidf.init(words_list1, update=False)
    testword = "我在九寨沟,很喜欢"
    #for word in jieba.cut(testword):
    #    print ('>>>>', word)
    pre = tfidf.predict(testword)
    print ('pre>>>>>', pre) 

    pre = tfidf._predict(testword)[0]
    print ('pre>>>>>', pre) 

```

3、训练w2v  (详见：)   ( python train_model/train_w2v.py )
```
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
```



4、训练bert  (详见：train_model/train_bert.py)
   (预训练权重下载：【百度网盘】链接:https://pan.baidu.com/s/1RVAHqL1CfLGltPWpoTyThw  密码:bp71)


5、训练albert  (详见：train_model/train_albert.py)
   (预训练权重下载：【百度网盘】链接:https://pan.baidu.com/s/1-PRsjQSwkGSpQkmjjnSXmw  密码:ynjs)


6、训练dssm  (详见：train_model/train_dssm.py)

