# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  faiss 测试
   Author :       machinelp
   Date :         2020-06-15
-------------------------------------------------

'''
import sys
import json 
import time
import faiss
import numpy as np
from faiss import normalize_L2
from textmatch.config.constant import Constant as const
from textmatch.core.text_embedding import TextEmbedding
from textmatch.tools.decomposition.pca import PCADecomposition
from textmatch.tools.faiss.faiss import FaissSearch

test_dict = {"id0": "其实事物发展有自己的潮流和规律",
   "id1": "当你身处潮流之中的时候，要紧紧抓住潮流的机会",
   "id2": "想办法脱颖而出，即使没有成功，也会更加洞悉时代的脉搏",
   "id3": "收获珍贵的知识和经验。而如果潮流已经退去",
   "id4": "这个时候再去往这个方向上努力，只会收获迷茫与压抑",
   "id5": "对时代、对自己都没有什么帮助",
   "id6": "但是时代的浪潮犹如海滩上的浪花，总是一浪接着一浪，只要你站在海边，身处这个行业之中，下一个浪潮很快又会到来。你需要敏感而又深刻地去观察，略去那些浮躁的泡沫，抓住真正潮流的机会，奋力一搏，不管成败，都不会遗憾。",
   "id7": "其实事物发展有自己的潮流和规律",
   "id8": "当你身处潮流之中的时候，要紧紧抓住潮流的机会" }


if __name__ == '__main__':
    # ['bow', 'tfidf', 'ngram_tfidf', 'bert']
    # ['bow', 'tfidf', 'ngram_tfidf', 'bert', 'w2v']
    # text_embedding = TextEmbedding( match_models=['bow', 'tfidf', 'ngram_tfidf', 'w2v'], words_dict=test_dict ) 
    text_embedding = TextEmbedding( match_models=['bow', 'tfidf', 'ngram_tfidf', 'w2v'], words_dict=None, update=False ) 
    feature_list = []
    for sentence in test_dict.values():
        pre = text_embedding.predict(sentence)
        feature = np.concatenate([pre[model] for model in ['bow', 'tfidf', 'ngram_tfidf', 'w2v']], axis=0)
        feature_list.append(feature)
    pca = PCADecomposition(n_components=8)
    data = np.array( feature_list )
    pca.fit( data )
    res = pca.transform( data )
    print('res>>', res)

   

    pre = text_embedding.predict("潮流和规律")
    feature = np.concatenate([pre[model] for model in ['bow', 'tfidf', 'ngram_tfidf', 'w2v']], axis=0)
    test = pca.transform( [feature] )

    faiss_search = FaissSearch( res, sport_mode=False )
    I, D = faiss_search.predict( test )
    '''
    faiss kmeans result times 8.0108642578125e-05
    I:[[0 7 3]]; D:[[0.7833399  0.7833399  0.63782495]]
    '''

    
    faiss_search = FaissSearch( res, sport_mode=True )
    I, D = faiss_search.predict( test )
    print( "I:{}; D:{}".format(I, D) )
    '''
    faiss kmeans result times 3.266334533691406e-05
    I:[[0 7 3]]; D:[[0.7833399  0.7833399  0.63782495]]
    '''
 


    '''
    d = 8                         # dimension
    #主要是为了测试不是归一化的vector
    training_vectors= res.astype('float32')
    normalize_L2(training_vectors)
    print('IndexFlatIP')
    index=faiss.IndexFlatIP(d)
    index.train(training_vectors)
    print(index)
    print('train')
    print(index.is_trained)
    print('add')
    print(index)
    index.nprobe = 20
    index.add(training_vectors)
    print('search')
    t1=time.time()
    D, I =index.search(training_vectors[:1], 3)
    t2 = time.time()
    print('faiss kmeans result times {}'.format(t2-t1))
    print(I[:3])                   # 表示最相近的前3个的index
    print(D[:3])                   # 表示最相近的前3个的相似度的值


    #加速版的cosine_similarity的计算
    d = 8                             # dimension
    training_vectors= res.astype('float32')
    normalize_L2(training_vectors)
    nlist = 5                         # 聚类中心的个数
    k = 3                             #邻居个数
    quantizer = faiss.IndexFlatIP(d)  # the other index，需要以其他index作为基础

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    # by default it performs inner-product search
    assert not index.is_trained
    index.train(training_vectors)
    assert index.is_trained
    index.nprobe = 20                  #300  # default nprobe is 1, try a few more
    index.add(training_vectors)       # add may be a bit slower as well
    t1=time.time()
    D, I = index.search(training_vectors[:1], k)  # actual search
    t2 = time.time()
    print('faiss kmeans result times {}'.format(t2-t1))
    # print(D[:k])  # neighbors of the k first queries
    print(I[:k])                   # 表示最相近的前3个的index
    print(D[:k])                   # 表示最相近的前3个的相似度的值
    '''

