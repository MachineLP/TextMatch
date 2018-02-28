# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""
'''
在进行训练之前要将训练数据筛选一下；
是不是为空；并且另存为jpg格式；
'''

import numpy as np  
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
#from vgg_preprocessing import*
import tensorflow as tf
# 适用于二级目录 。。。/图片类别文件/图片（.png ,jpg等）


#input = tf.placeholder(tf.float32, [None, None, 3])  
#out = preprocess_image(input, 224,224, is_training=False)
#out = preprocess_image(input, 224,224, is_training=True)
#sess = tf.Session()
def load_img(imgDir,imgFoldName, img_label):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = []#np.empty((imgNum,224,224,3),dtype="float32")
    label = []#np.empty((imgNum,),dtype="uint8")
    for i in range (imgNum):
        img = cv2.imread(imgDir+imgFoldName+"/"+imgs[i])
        #for j in range(1):
        if img is not None:
            #img_ = sess.run(out, feed_dict={input: img})
            img_ = img
            # img_ = cv2.resize(img_, (960, 540))
            #save_path = "train/"+imgFoldName+"/"+imgs[i]
            save_path = dir_path+imgFoldName+"/"+imgs[i]
            save_path = save_path.split('.')[0]
            #save_path = save_path + str(j) + '.jpg'
            save_path = save_path + '.jpg'
            print (save_path)
            cv2.imwrite(save_path, img_)
        '''img = cv2.resize(img, (224, 224))  
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        # label[i] = int(imgs[i].split('.')[0])
        label[i] = int(img_label)'''
    return data,label
'''
craterDir = "train/"
foldName = "0male"
data, label = load_Img(craterDir,foldName, 0)

print (data[0].shape)
print (label[0])'''

def load_database(imgDir):
    img_path = os.listdir(imgDir)
    train_imgs = []
    train_labels = []
    for i, path in enumerate(img_path):
        craterDir = imgDir + '/'
        foldName = path
        data, label = load_img(craterDir,foldName, i)
        train_imgs.extend(data)
        train_labels.extend(label)
    #打乱数据集
    index = [i for i in range(len(train_imgs))]    
    np.random.shuffle(index)   
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]  
    train_labels = train_labels[index] 
    return train_imgs, train_labels


def get_next_batch(train_imgs, train_labels, pointer, batch_size=64):
    batch_x = np.zeros([batch_size, 224,224,3])  
    batch_y = np.zeros([batch_size, ]) 
    for i in range(batch_size):  
        #image = cv2.imread(image_path[i+pointer*batch_size])
        #image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))  
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        #image = Image.open(image_path[i+pointer*batch_size])
        #image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  
        #image = image.convert('L')  
        #大神说，转成数组再搞
        #image=np.array(image)
        #
        image = train_imgs[i+pointer*batch_size]
        '''
        m = image.mean()
        s = image.std()
        min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
        std = max(min_s, s)
        image = (image-m)/std'''
        image = (image-127.5)
        
        batch_x[i,:,:,:] = image
        # print labels[i+pointer*batch_size]
        batch_y[i] = train_labels[i+pointer*batch_size]
    return batch_x, batch_y


def test():

    craterDir = "gender"
    global dir_path
    dir_path = "train_gender/"
    #dir_path = "train/"
    data, label = load_database(craterDir)
    #dir = "/1female"
    #data, label = load_img(craterDir,dir,0)
    print (data.shape)
    print (len(data))
    print (data[0].shape)
    print (label[0])


if __name__ == '__main__':
    test()
