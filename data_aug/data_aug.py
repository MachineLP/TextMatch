
# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np  
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
from skimage import exposure

class data_aug(object):
    
    def __init__(self, img):
        self.image= img
    
    # 左右镜像
    def _random_fliplr(self, random_fliplr = True):
        if random_fliplr and np.random.choice([True, False]):
            self.image = np.fliplr(self.image) # 左右

    # 上下镜像    
    def _random_flipud(self, random_flipud = True):
        if random_flipud and np.random.choice([True, False]):
            self.image = np.flipud(self.image) # 上下
        
    # 改变光照
    def _random_exposure(self, random_exposure = True):
        if random_exposure and np.random.choice([True, False]):
            e_rate = np.random.uniform(0.5,1.5)
            self.image = exposure.adjust_gamma(self.image, e_rate)
        
    # 旋转
    def _random_rotation(self, random_rotation = True):
        if random_rotation and np.random.choice([True, False]):
            w,h = self.image.shape[1], self.image.shape[0]
            # 0-180随机产生旋转角度。
            angle = np.random.randint(0,10)
            RotateMatrix = cv2.getRotationMatrix2D(center=(w/2, h/2), angle=angle, scale=0.7)
            # image = cv2.warpAffine(image, RotateMatrix, (w,h), borderValue=(129,137,130))
            self.image = cv2.warpAffine(self.image, RotateMatrix, (w,h), borderMode=cv2.BORDER_REPLICATE)
    
    # 裁剪
    def _random_crop(self, crop_size = 299, random_crop = True):
        if random_crop and np.random.choice([True, False]):
            if self.image.shape[1] > crop_size:
                sz1 = self.image.shape[1] // 2
                sz2 = crop_size // 2
                diff = sz1 - sz2
                (h, v) = (np.random.randint(0, diff + 1), np.random.randint(0, diff + 1))
                self.image = self.image[v:(v + crop_size), h:(h + crop_size), :]
    # 
    def get_aug_img(self):
        data_aug_list = [self._random_fliplr, self._random_flipud, self._random_rotation, self._random_exposure, self._random_crop]
        data_aug_func = np.random.choice(data_aug_list, 2)
        for func in data_aug_func:
            func()
        return self.image
    

