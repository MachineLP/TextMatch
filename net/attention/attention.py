# coding=utf-8

import tensorflow as tf

def attention(inputs, attention_size, time_major=False):  
    if isinstance(inputs, tuple):  
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.  
        inputs = tf.concat(inputs, 2)  
  
    if time_major:  
        # (T,B,D) => (B,T,D)  
        inputs = tf.transpose(inputs, [1, 0, 2])  
  
    inputs_shape = inputs.shape  
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer  
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer  
  
    # Attention mechanism  
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))  
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  
  
    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))  
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))  
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])  
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  
  
    # Output of Bi-RNN is reduced with attention vector  
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)  
  
    return output  
