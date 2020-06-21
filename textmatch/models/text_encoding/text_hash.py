# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  text hash实现
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''


import os
import hashlib
from simhash import Simhash

class TextEncoding():
    def __init__(self):
        pass

    def get_hash(self, words):
        md5 = hashlib.md5()
        md5.update(words.encode("utf-8"))
        return md5.hexdigest()

    def get_simhash(self, text, f_num=64):
        simhash_value = Simhash(text, f=f_num)  
        return simhash_value

    def get_sim_simhash(self, text1, text2, f_num=64):
        a_simhash = Simhash(text1, f=f_num)  
        b_simhash = Simhash(text2, f=f_num)
        max_hashbit = max(len(bin(a_simhash.value)), len(bin(b_simhash.value)))
        distance = a_simhash.distance(b_simhash)
        sim = 1 - distance / max_hashbit  
        return sim
    
    def get_sim_from_simhash(self, a_simhash, b_simhash):
        max_hashbit = max(len(bin(a_simhash.value)), len(bin(b_simhash.value)))
        distance = a_simhash.distance(b_simhash)
        sim = 1 - distance / max_hashbit 
        return sim




