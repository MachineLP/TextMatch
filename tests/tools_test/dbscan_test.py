# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  DBSCAN 测试
   Author :       machinelp
   Date :         2020-06-15
-------------------------------------------------

'''

import sys
import json 
import numpy as np
from textmatch.config.constant import Constant as const
from textmatch.core.text_embedding import TextEmbedding
from textmatch.tools.clustering.dbscan import DBSCANClustering
from textmatch.tools.decomposition.pca import PCADecomposition

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
    text_embedding = TextEmbedding( match_models=['bow', 'tfidf', 'ngram_tfidf', 'w2v'], words_dict=test_dict ) 
    km = DBSCANClustering(eps=0.5, min_samples=2)
    # pre = text_embedding.predict( "其实事物发展有自己的潮流和规律" ) 
    feature_list = []
    for sentence in test_dict.values():
        pre = text_embedding.predict(sentence)
        feature = np.concatenate([pre[model] for model in ['bow', 'tfidf', 'ngram_tfidf', 'w2v']], axis=0)
        feature_list.append(feature)
    # print ('feature_list:', feature_list) 
    label_list = km.predict(feature_list)
    print ('label_list:', label_list) 
    clustering_dict = dict()
    for label, sentence in zip(label_list, test_dict.values()):
        key = str(label)
        if label not in clustering_dict.keys():
            clustering_dict[key] = [sentence]
        else:
            clustering_dict[key] = clustering_dict[key].append(sentence)
    print('clustering_dict>>', clustering_dict)


    pca = PCADecomposition(n_components=4)
    data = np.array( feature_list )
    pca.fit( data )
    res = pca.transform( data )
    label_list = km.predict(res)
    print ('label_list:', label_list) 
    clustering_dict = dict()
    for label, sentence in zip(label_list, test_dict.values()):
        key = str(label)
        if label not in clustering_dict.keys():
            clustering_dict[key] = [sentence]
        else:
            clustering_dict[key] = clustering_dict[key].append(sentence)
    print('clustering_dict>>', clustering_dict)




