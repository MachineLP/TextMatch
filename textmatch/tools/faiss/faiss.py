# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  Faiss
   Author :       machinelp
   Date :         2020-06-15
-------------------------------------------------

'''

import time
import faiss
import numpy as np
from faiss import normalize_L2
from textmatch.config.config import cfg


class FaissSearch():
   def __init__(self, data_vector=None, sport_mode=True):
      self.data_vector = data_vector
      self.d = self.data_vector.shape[1]
      if self.data_vector is None:
         self.data_vector = self._load_dataset() 
      self._init( sport_mode )

   def _load_dataset(self):
      pass

   def _normalize(self, vector):
      normalize_L2( vector )
      return vector

   def _init(self, sport_mode, nlist=cfg.faiss.n_clusters, nprobe=cfg.faiss.nprobe):
      train_vector = self.data_vector.astype('float32')
      train_vector = self._normalize( train_vector )
      if sport_mode:
         # nlist = 3                               # 聚类中心的个数
         quantizer = faiss.IndexFlatIP( self.d )  # the other index，需要以其他index作为基础
         self.index = faiss.IndexIVFFlat(quantizer, self.d, nlist, faiss.METRIC_INNER_PRODUCT)
         # by default it performs inner-product search
         assert not self.index.is_trained
         self.index.train( train_vector )
         assert self.index.is_trained
         self.index.nprobe = nprobe                # default nprobe is 1, try a few more
         self.index.add( train_vector )       
      else:
         self.index = faiss.IndexFlatIP( self.d )
         self.index.train( train_vector )
         assert self.index.is_trained
         self.index.add( train_vector )

   def predict(self, vector, topn=3):
      vector = self._normalize( vector.astype('float32') )
      t1=time.time()
      D, I = self.index.search(vector, topn)
      t2 = time.time()
      print('faiss kmeans result times {}'.format(t2-t1))
      #print(I[:topn])                   # 表示最相近的前3个的index
      #print(D[:topn])                   # 表示最相近的前3个的相似度的值
      return I, D



