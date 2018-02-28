# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng"""

import numpy as np  
import numpy as np
import os
from PIL import Image
import cv2

import csv
import argparse, json, textwrap
import sys
import csv


def result2num(out, image_path): 
    # print (out)
    dc = out[image_path]
    # print (dc)

    if dc.get("help", ""):
        print ("help is true!")
        dc.pop('help')
        print (">>>>>>>>", dc)

    def dict2list(dic:dict):  
        ''''' 将字典转化为列表 '''  
        keys = dic.keys()  
        vals = dic.values()  
        lst = [(key, val) for key, val in zip(keys, vals)]  
        return lst 
    dc = sorted(dict2list(dc), key=lambda d:d[1], reverse=True)
    # print (dc[0][0])
    if dc[0][0] == 'NG1':
        return 0
    if dc[0][0] == 'NG2':
        return 1
    if dc[0][0] == 'OK':
        return 2

file = open("output.csv", "r")

err_num = 0
sample_num = 0
for r in file:
    sample_num = sample_num + 1
    # 转为字典
    r = eval(r)
    # 转为列表
    image_path = list(r.keys())
    la = 888888888888888
    label = str (str(image_path[0]).split('/')[1])
    # print (label)
    if label == 'NG1':
        la = 0
    if label == 'NG2':
        la = 1
    if label == 'OK':
        la = 2
    print (la)

    image_path = str(image_path[0])
    res = result2num(r, image_path)
    print (res)

    if (la != res):
        err_num = err_num + 1
print (sample_num)
print (err_num)
acc_num = sample_num - err_num
print ('accuracy >>> ', acc_num/sample_num)
