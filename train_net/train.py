# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import argparse
import os
from PIL import Image
from datetime import datetime
import math
import time
import cv2

from keras.utils import np_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from load_image import get_next_batch_from_path, shuffle_train_data
except:
    from load_image.load_image import get_next_batch_from_path, shuffle_train_data
from load_image import load_image

# net_arch
from net.z_build_net import build_net



def g_parameter(checkpoint_exclude_scopes):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    print (exclusions)
    # 需要加载的参数。
    variables_to_restore = []
    # 需要训练的参数
    variables_to_train = []
    for var in slim.get_model_variables():
    # 切记不要用下边这个，这是个天大的bug，调试了3天。
    # for var in tf.trainable_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                variables_to_train.append(var)
                print ("ok")
                print (var.op.name)
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore,variables_to_train


def train(train_data,train_label,valid_data,valid_label,train_n,valid_n,IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,EARLY_STOP_PATIENCE,batch_size=64,keep_prob=0.8,
           arch_model="arch_inception_v4",checkpoint_exclude_scopes="Logits_out", checkpoint_path="pretrain/inception_v4/inception_v4.ckpt"):

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    #Y = tf.placeholder(tf.float32, [None, 4])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    is_training = tf.placeholder(tf.bool, name='is_training')
    k_prob = tf.placeholder(tf.float32) # dropout

    # 定义模型
    net = build_net.net_arch()
    if arch_model == "arch_inception_v4":
        net = net.arch_inception_v4(X, num_classes, k_prob, is_training)
    elif arch_model == "arch_resnet_v2_50":
        net = net.arch_resnet_v2_50(X, num_classes, k_prob, is_training)
    elif arch_model == "vgg_16":
        net = net.arch_vgg16(X, num_classes, k_prob, is_training)
    elif arch_model == "arch_inception_v4_rnn":
        net = net.arch_inception_v4_rnn(X, num_classes, k_prob, is_training)
    elif arch_model == "arch_inception_v4_rnn_attention":
        net = net.arch_inception_v4_rnn_attention(X, num_classes, k_prob, is_training)

    # 
    variables_to_restore,variables_to_train = g_parameter(checkpoint_exclude_scopes)

    # loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = net))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = net))

    var_list = variables_to_train
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=var_list)
    predict = tf.reshape(net, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(Y, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # tensorboard
    with tf.name_scope('tmp/'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()
    #------------------------------------------------------------------------------------#
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    #
    log_dir = arch_model + '_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    saver2 = tf.train.Saver(tf.global_variables())
    model_path = 'model/fine-tune'

    net_vars = variables_to_restore
    saver_net = tf.train.Saver(net_vars)
    # checkpoint_path = 'pretrain/inception_v4.ckpt'
    saver_net.restore(sess, checkpoint_path)

    # early stopping
    best_valid = np.inf
    best_valid_epoch = 0
    
    # saver2.restore(sess, "model/fine-tune-1120")
    for epoch_i in range(epoch):
        for batch_i in range(int(train_n/batch_size)):

            images_train, labels_train = get_next_batch_from_path(train_data, train_label, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=True)

            los, _ = sess.run([loss,optimizer], feed_dict={X: images_train, Y: labels_train, k_prob:keep_prob, is_training:True})
            # print (los)

            if batch_i % 100 == 0:
                images_valid, labels_valid = get_next_batch_from_path(valid_data, valid_label, batch_i%(int(valid_n/batch_size)), IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=False)
                ls, acc = sess.run([loss, accuracy], feed_dict={X: images_valid, Y: labels_valid, k_prob:1.0, is_training:False})
                print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, ls, acc))
                #if acc > 0.90:
                #    saver2.save(sess, model_path, global_step=batch_i, write_meta_graph=False)
            elif batch_i % 20 == 0:
                loss_, acc_, summary_str = sess.run([loss, accuracy, summary_op], feed_dict={X: images_train, Y: labels_train, k_prob:1.0, is_training:False})
                writer.add_summary(summary_str, global_step=((int(train_n/batch_size))*epoch_i+batch_i))
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss_, acc_))
                 
        print('Epoch===================================>: {:>2}'.format(epoch_i))
        valid_ls = 0
        valid_acc = 0
        for batch_i in range(int(valid_n/batch_size)):
            images_valid, labels_valid = get_next_batch_from_path(valid_data, valid_label, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=False)
            epoch_ls, epoch_acc = sess.run([loss, accuracy], feed_dict={X: images_valid, Y: labels_valid, k_prob:1.0, is_training:False})
            valid_ls = valid_ls + epoch_ls
            valid_acc = valid_acc + epoch_acc
        print('Epoch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(epoch_i, valid_ls/int(valid_n/batch_size), valid_acc/int(valid_n/batch_size)))
        if valid_acc/int(valid_n/batch_size) > 0.90:
            saver2.save(sess, model_path, global_step=epoch_i, write_meta_graph=False)
        loss_valid = valid_ls/int(valid_n/batch_size)
        if loss_valid < best_valid:
            best_valid = loss_valid
            best_valid_epoch = epoch_i
        elif best_valid_epoch + EARLY_STOP_PATIENCE < epoch_i:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(best_valid, best_valid_epoch))
            break
        # if valid_acc/int(valid_n/batch_size) > 0.90:
        #     saver2.save(sess, model_path, global_step=epoch_i, write_meta_graph=False)

        print('>>>>>>>>>>>>>>>>>>>shuffle train_data<<<<<<<<<<<<<<<<<')
        # 每个epoch，重新打乱一次训练集：
        train_data, train_label = shuffle_train_data(train_data, train_label)
    writer.close()       
    sess.close()

if __name__ == '__main__':

    IMAGE_HEIGHT = 299
    IMAGE_WIDTH = 299
    num_classes = 4
    # epoch
    epoch = 100
    batch_size = 16
    # 模型的学习率
    learning_rate = 0.00001
    keep_prob = 0.8

    
    ##----------------------------------------------------------------------------##
    # 设置训练样本的占总样本的比例：
    train_rate = 0.9
    # 每个类别保存到一个文件中，放在此目录下，只要是二级目录就可以。
    craterDir = "train"
    # arch_model="arch_inception_v4";  arch_model="arch_resnet_v2_50"; arch_model="vgg_16"
    arch_model="arch_inception_v4"
    checkpoint_exclude_scopes = "Logits_out"
    checkpoint_path="pretrain/inception_v4/inception_v4.ckpt"
    

    ##----------------------------------------------------------------------------##
    print ("-----------------------------load_image.py start--------------------------")
    # 准备训练数据
    all_image = load_image.load_image(craterDir, train_rate)
    train_data, train_label, valid_data, valid_label= all_image.gen_train_valid_image()
    image_n = all_image.image_n
    # 样本的总数量
    print ("样本的总数量:")
    print (image_n)
    # 定义90%作为训练样本
    train_n = all_image.train_n
    valid_n = all_image.valid_n
    # ont-hot
    train_label = np_utils.to_categorical(train_label, num_classes)
    valid_label = np_utils.to_categorical(valid_label, num_classes)
    ##----------------------------------------------------------------------------##

    print ("-----------------------------train.py start--------------------------")
    train(train_data,train_label,valid_data,valid_label,train_n,valid_n,IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,batch_size,keep_prob,
          arch_model, checkpoint_exclude_scopes, checkpoint_path)
