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
from data_aug import data_aug

# 适用于二级目录 。。。/图片类别文件/图片（.png ,jpg等）

class load_image(object):
    
    def __init__(self, img_dir, train_rate):
        self.imgDir = img_dir
        self.train_rate = train_rate
    
    def _load_img_path(self, img_label):
        imgs = os.listdir(os.path.join(self.craterDir, self.foldName))  
        imgNum = len(imgs)
        self.data = []
        self.label = []
        for i in range (imgNum):
            img_path = os.path.join(self.craterDir, self.foldName, imgs[i]) 
            # 用来检测图片是否有效，放在这里会太费时间。
            # img = cv2.imread(img_path)
            # if img is not None:
            self.data.append(img_path)
            self.label.append(int(img_label))

    def _shuffle_train_data(self):
        index = [i for i in range(len(self.train_imgs))]
        np.random.shuffle(index)
        self.train_imgs = np.asarray(self.train_imgs)
        self.train_labels = np.asarray(self.train_labels)
        self.train_imgs = self.train_imgs[index]
        self.train_labels = self.train_labels[index]


    def _load_database_path(self):
        img_path = os.listdir(self.imgDir)
        self.train_imgs = []
        self.train_labels = []
        for i, path in enumerate(img_path):
            self.craterDir = self.imgDir
            self.foldName = path
            self._load_img_path(i)
            self.train_imgs.extend(self.data)
            self.train_labels.extend(self.label)
            print ("文件名对应的label:")
            print (path, i)
        #打乱数据集
        self._shuffle_train_data()
        # 数据集的数量
        self.image_n = len(self.train_imgs)
    
    def gen_train_valid_image(self):
        self._load_database_path()
        self.train_n = int(self.image_n * self.train_rate)
        self.valid_n = int(self.image_n * (1 - self.train_rate))
        return self.train_imgs[0:self.train_n], self.train_labels[0:self.train_n], self.train_imgs[self.train_n:self.image_n], self.train_labels[self.train_n:self.image_n]

    '''
    def get_next_batch_from_path(image_path, image_labels, pointer, IMAGE_HEIGHT=299, IMAGE_WIDTH=299, batch_size=64, is_train=True):
        batch_x = np.zeros([batch_size, IMAGE_HEIGHT,IMAGE_WIDTH,3])
        num_classes = len(image_labels[0])
        batch_y = np.zeros([batch_size, num_classes]) 
        for i in range(batch_size):  
            image = cv2.imread(image_path[i+pointer*batch_size])
            image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
            if is_train:
                img_aug = data_aug.data_aug(image)
                image = img_aug.get_aug_img()
            # image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))  
            # 选择自己预处理方式：
            
            # m = image.mean()
            # s = image.std()
            # min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
            # std = max(min_s, s)
            # image = (image-m)/std
            # image = (image-127.5)
            image = image / 255.0
            image = image - 0.5
            image = image * 2
        
            batch_x[i,:,:,:] = image
            # print labels[i+pointer*batch_size]
            batch_y[i] = image_labels[i+pointer*batch_size]
        return batch_x, batch_y'''


def shuffle_train_data(train_imgs, train_labels):
    index = [i for i in range(len(train_imgs))]
    np.random.shuffle(index)
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]
    train_labels = train_labels[index]
    return train_imgs, train_labels

#------------------------------------------------#
# 功能：按照图像最小的边进行缩放
# 输入：img：图像，resize_size：需要的缩放大小
# 输出：缩放后的图像
#------------------------------------------------#
def img_crop_pre(img, resize_size=336):
    h, w, _ = img.shape
    deta = h if h < w else w
    alpha = resize_size / float(deta)
    # print (alpha)
    img = cv2.resize(img, (int(h*alpha), int(w*alpha)))
    return img

def get_next_batch_from_path(image_path, image_labels, pointer, IMAGE_HEIGHT=299, IMAGE_WIDTH=299, batch_size=64, is_train=True):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT,IMAGE_WIDTH,3])
    num_classes = len(image_labels[0])
    batch_y = np.zeros([batch_size, num_classes]) 
    for i in range(batch_size):  
        image = cv2.imread(image_path[i+pointer*batch_size])
        image = img_crop_pre(image)
        if is_train:
            img_aug = data_aug.data_aug(image)
            image = img_aug.get_aug_img()
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))  
        # 选择自己预处理方式：
        '''
        m = image.mean()
        s = image.std()
        min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
        std = max(min_s, s)
        image = (image-m)/std'''
        # image = (image-127.5)
        image = image / 255.0
        image = image - 0.5
        image = image * 2
        
        batch_x[i,:,:,:] = image
        # print labels[i+pointer*batch_size]
        batch_y[i] = image_labels[i+pointer*batch_size]
    return batch_x, batch_y


def test():

    craterDir = "train"
    data, label = load_database(craterDir)
    print (data.shape)
    print (len(data))
    print (data[0].shape)
    print (label[0])
    batch_x, batch_y = get_next_batch_from_path(data, label, 0, IMAGE_HEIGHT=299, IMAGE_WIDTH=299, batch_size=64, is_train=True)
    print (batch_x)
    print (batch_y)

if __name__ == '__main__':
    test()

