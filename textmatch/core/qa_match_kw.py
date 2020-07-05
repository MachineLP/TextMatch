# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QAMatchKW 
   Author :       machinelp
   Date :         2020-07-05
-------------------------------------------------

'''
import re
import json
import numpy as np


class QAMatchKW():
    '''
        output: {'id0':0.2, 'id1':0.5, ...}
    '''

    def __init__(self, qkw_path=None, akw_path=None, pretrain=False):
        self.a_flag = a_flag
        self.qkw_dict = self._load_json(qkw_path)
        self.akw_path = self._load_json(akw_path)

    def _load_json(self, json_path):
        with open(json_path, 'r', encoding='UTF-8') as f:
            qkw_json = json.load(f)
        return qkw_json

    def _dump_qkw(self, qakw_dict, qkw_pretrain_path):
        # 保存到json文件
        with open(qkw_pretrain_path, 'w', encoding='utf-8') as w:
            json.dump(qakw_dict, w, ensure_ascii=False)

    # 关键词表中的多个关键词都要满足的。
    def get_qkw_dict(self):
        return self.qkw_dict
    def get_akw_dict(self):
        return self.akw_path

    def pp_and(self):
        pass

    def pp_or(self):
        pass
    
    # words: 兰蔻的生产日期   res_dict:{'id123':0.8}  score_threshold:0.7
    # kw_dict: {'id123':['兰蔻', '生产日期'], 'id124':['兰蔻', '眼霜', '用法|使用方法']}
    # 主要针对Q 
    def post_processing_q(self, words, res_dict, score_threshold=0.7, filter_keys=None, sep='/', sep_re='|'):
        kw_dict = self.qkw_dict  
        res_post_pro = {}
        for res_key, res_value in res_dict.items():
            if (filter_keys!=None) and (res_key not in filter_keys):
                continue
            kw_value = kw_dict.get(res_key, [])
            if kw_value ==[]:
                if res_value >= score_threshold :
                    res_post_pro[res_key] = res_value
            else:
                hit_flag = 0
                for per_kw_value in kw_value:
                    q_kw = per_kw_value.replace(sep, sep_re)   #取关键词, 替换成正则需要的形式
                    pattern = re.compile(q_kw)
                    if len(re.findall(pattern, words))>0:
                        hit_flag += 1
                if len(kw_value) == hit_flag:    
                    res_post_pro[res_key] = 1.0 #res_value
        return res_post_pro
    
    
    def _to_value(self, value):
        if isinstance(value, list):
            return value[0]
        return value

    def _to_kw(self, value):
        if isinstance(value, list):
            return value[1]
        return ""
    
    # 主要针对A
    # 答句中出现过所有关键词score为1； 问句中没有出现过关键词score为0。
    # kw_dict: {'id123':['九月十八|9月18']
    def post_processing_a(self, words, res_dict, sep='/', sep_re='|'):
        kw_dict = self.akw_path
        for res_key, res_value in res_dict.items():
            kw_value = kw_dict.get(res_key, [])
            if kw_value!=[] and kw_value != None:
                hit_flag = 0
                for per_kw_value in kw_value:
                    a_kw = per_kw_value.replace(sep, sep_re)   #取关键词, 替换成正则需要的形式
                    pattern = re.compile(a_kw)
                    if len(set(re.findall(pattern, words))) > 0:
                        hit_flag += 1
                if len(kw_value) == hit_flag:    
                    res_dict[res_key] = [1.0, a_kw]
                else:
                    res_dict[res_key] = [0.0, a_kw] 
            else:
                res_dict[res_key] = [self._to_value(res_value), self._to_kw(res_value)]
        return res_dict

