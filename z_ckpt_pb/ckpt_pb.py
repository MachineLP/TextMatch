# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""


import tensorflow as tf
slim = tf.contrib.slim
import os.path
import argparse
from tensorflow.python.framework import graph_util
from inception_v4 import *
from inception_preprocessing import *



MODEL_DIR = "model/"
MODEL_NAME = "frozen_model.pb"

if not tf.gfile.Exists(MODEL_DIR): #创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

batch_size = 32
height, width = 299, 299
num_classes = 3
X = tf.placeholder(tf.float32, [None, height, width, 3], name = "inputs_placeholder")  
'''
X = tf.placeholder(tf.uint8, [None, None, 3],name = "inputs_placeholder")
X = tf.image.encode_jpeg(X, format='rgb')  # 单通道用  'grayscale'
X = tf.image.decode_jpeg(X, channels=3)
X = preprocess_for_eval(X, 299,299)
X = tf.reshape(X, [-1,299,299,3])'''
Y = tf.placeholder(tf.float32, [None, num_classes])  
#keep_prob = tf.placeholder(tf.float32) # dropout
#keep_prob_fc = tf.placeholder(tf.float32) # dropout
arg_scope = inception_v4_arg_scope()
with slim.arg_scope(arg_scope):
    net, end_points = inception_v4(X, is_training=False)
#sess1 = tf.Session()
#saver1 = tf.train.Saver(tf.global_variables())
#checkpoint_path = 'model/inception_v4.ckpt'
#saver1.restore(sess1, checkpoint_path)
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
    with tf.variable_scope('Logits_out'):
        # 8 x 8 x 1536
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                  scope='AvgPool_1a_out')
        # 1 x 1 x 1536
        dropout_keep_prob = 1.0
        net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out')
        net = slim.flatten(net, scope='PreLogitsFlatten_out')
        # 1536
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
        net = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits_out1')
# net = tf.nn.softmax(net)
net = tf.nn.sigmoid(net)
predict = tf.reshape(net, [-1, num_classes], name='predictions')

for var in tf.trainable_variables():
    print (var.op.name)


def freeze_graph(model_folder):
    #checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    #input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    input_checkpoint = model_folder
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME) #PB模型保存路径

    output_node_names = "predictions" #原模型输出操作节点的名字
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) #得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.
    saver = tf.train.Saver()

    graph = tf.get_default_graph() #获得默认的图
    input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, input_checkpoint) #恢复图并得到数据

        #print "predictions : ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]}) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字

        output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",") #如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        for op in graph.get_operations():
            #print(op.name, op.values())
            print("name:",op.name)
        print ("success!")


        #下面是用于测试， 读取pd模型，答应每个变量的名字。
        graph = load_graph("model/frozen_model.pb")
        for op in graph.get_operations():
            #print(op.name, op.values())
            print("name111111111111:",op.name)
        pred = graph.get_tensor_by_name('prefix/inputs_placeholder:0')
        print (pred)
        temp = graph.get_tensor_by_name('prefix/predictions:0')
        print (temp)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", type=str, help="input ckpt model dir", default="model/cnn_model-1700") #命令行解析，help是提示符，type是输入的类型，
    # 这里运行程序时需要带上模型ckpt的路径，不然会报 error: too few arguments
    aggs = parser.parse_args()
    freeze_graph(aggs.model_folder)
    # freeze_graph("model/ckpt") #模型目录
# python ckpt_pb.py "model/fine-tune-160"
